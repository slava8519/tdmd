#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def fcc_points(*, box: float, lattice_a: float) -> np.ndarray:
    n = int(np.ceil(float(box) / float(lattice_a)))
    basis = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        dtype=float,
    )
    out: list[np.ndarray] = []
    for ix in range(n):
        ox = float(ix) * float(lattice_a)
        for iy in range(n):
            oy = float(iy) * float(lattice_a)
            for iz in range(n):
                oz = float(iz) * float(lattice_a)
                pts = (basis * float(lattice_a)) + np.asarray([ox, oy, oz], dtype=float)
                mask = np.all(pts < float(box), axis=1)
                if np.any(mask):
                    out.append(pts[mask])
    if not out:
        return np.empty((0, 3), dtype=float)
    return np.vstack(out)


def points_in_box(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    return np.all((pts >= np.asarray(lo, dtype=float)) & (pts <= np.asarray(hi, dtype=float)), axis=1)


def build_al_crack_state(
    *,
    target_atoms: int,
    box: float,
    lattice_a: float,
    seed: int,
    velocity_std: float,
    crack_lo_frac: tuple[float, float, float] = (0.15, 0.36, 0.02),
    crack_hi_frac: tuple[float, float, float] = (0.85, 0.64, 0.56),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    target = int(target_atoms)
    if target <= 0:
        raise ValueError("target_atoms must be positive")
    if float(box) <= 0.0:
        raise ValueError("box must be positive")

    al_all = fcc_points(box=float(box), lattice_a=float(lattice_a))
    crack_lo = np.asarray(
        [
            float(crack_lo_frac[0]) * float(box),
            float(crack_lo_frac[1]) * float(box),
            float(crack_lo_frac[2]) * float(box),
        ],
        dtype=float,
    )
    crack_hi = np.asarray(
        [
            float(crack_hi_frac[0]) * float(box),
            float(crack_hi_frac[1]) * float(box),
            float(crack_hi_frac[2]) * float(box),
        ],
        dtype=float,
    )
    keep_al = ~points_in_box(al_all, crack_lo, crack_hi)
    al_after_crack = np.asarray(al_all[keep_al], dtype=float)
    available_after_crack = int(al_after_crack.shape[0])
    if available_after_crack < target:
        raise RuntimeError(
            f"crack removed too many atoms for target={target}: available_after_crack={available_after_crack}"
        )

    rng = np.random.default_rng(int(seed))
    selected = rng.permutation(available_after_crack)[:target]
    positions = np.asarray(al_after_crack[selected], dtype=float).copy()
    types = np.ones((target,), dtype=np.int32)
    masses = np.full((target,), 26.9815385, dtype=float)
    velocities = rng.normal(0.0, float(velocity_std), size=(target, 3)).astype(float)
    velocities -= np.mean(velocities, axis=0, keepdims=True)
    return positions, types, masses, velocities, crack_lo, crack_hi, available_after_crack


def write_al_crack_task_yaml(
    *,
    out_path: Path,
    box: float,
    dt: float,
    steps: int,
    cutoff: float,
    positions: np.ndarray,
    types: np.ndarray,
    masses: np.ndarray,
    velocities: np.ndarray,
    eam_file: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("task_version: 1\n")
        f.write("units: metal\n")
        f.write("box:\n")
        f.write(f"  x: {float(box):.6f}\n")
        f.write(f"  y: {float(box):.6f}\n")
        f.write(f"  z: {float(box):.6f}\n")
        f.write("  pbc: [true, true, true]\n")
        f.write("atoms:\n")
        for idx in range(int(positions.shape[0])):
            rid = idx + 1
            rx, ry, rz = np.asarray(positions[idx], dtype=float)
            vx, vy, vz = np.asarray(velocities[idx], dtype=float)
            f.write(f"  - id: {rid}\n")
            f.write(f"    type: {int(types[idx])}\n")
            f.write(f"    mass: {float(masses[idx]):.7f}\n")
            f.write(f"    r: [{rx:.6f}, {ry:.6f}, {rz:.6f}]\n")
            f.write(f"    v: [{vx:.6f}, {vy:.6f}, {vz:.6f}]\n")
        f.write("potential:\n")
        f.write("  kind: eam/alloy\n")
        f.write("  params:\n")
        f.write(f"    file: {str(eam_file)}\n")
        f.write("    elements: [Al]\n")
        f.write(f"cutoff: {float(cutoff):.6f}\n")
        f.write(f"dt: {float(dt):.6f}\n")
        f.write(f"steps: {int(steps)}\n")
        f.write("ensemble:\n")
        f.write("  kind: nve\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a pure-Al microcrack task with exact atom count")
    ap.add_argument("--out", default="results/task_al_crack_100k_nve100.yaml")
    ap.add_argument("--target-atoms", type=int, default=100000)
    ap.add_argument("--box", type=float, default=122.0)
    ap.add_argument("--lattice-a", type=float, default=4.05)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cutoff", type=float, default=6.5)
    ap.add_argument(
        "--eam-file",
        default="examples/potentials/eam_alloy/Al_zhou.eam.alloy",
        help="Single-element Al setfl file",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--velocity-std", type=float, default=0.020)
    args = ap.parse_args()

    positions, types, masses, velocities, crack_lo, crack_hi, available_after_crack = build_al_crack_state(
        target_atoms=int(args.target_atoms),
        box=float(args.box),
        lattice_a=float(args.lattice_a),
        seed=int(args.seed),
        velocity_std=float(args.velocity_std),
    )
    out_path = Path(args.out)
    write_al_crack_task_yaml(
        out_path=out_path,
        box=float(args.box),
        dt=float(args.dt),
        steps=int(args.steps),
        cutoff=float(args.cutoff),
        positions=positions,
        types=types,
        masses=masses,
        velocities=velocities,
        eam_file=str(args.eam_file),
    )

    print(f"[ok] wrote {out_path}")
    print(
        f"[summary] total={int(args.target_atoms)} available_after_crack={available_after_crack} "
        f"box={float(args.box):.3f} cutoff={float(args.cutoff):.3f}"
    )
    print(
        f"[crack] lo=({crack_lo[0]:.3f},{crack_lo[1]:.3f},{crack_lo[2]:.3f}) "
        f"hi=({crack_hi[0]:.3f},{crack_hi[1]:.3f},{crack_hi[2]:.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
