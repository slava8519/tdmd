#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def _fcc_points(box: float, a: float) -> np.ndarray:
    n = int(np.ceil(box / a))
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
        ox = float(ix) * a
        for iy in range(n):
            oy = float(iy) * a
            for iz in range(n):
                oz = float(iz) * a
                pts = (basis * a) + np.asarray([ox, oy, oz], dtype=float)
                m = np.all(pts < box, axis=1)
                if np.any(m):
                    out.append(pts[m])
    if not out:
        return np.empty((0, 3), dtype=float)
    return np.vstack(out)


def _in_box(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.all((points >= lo) & (points <= hi), axis=1)


def _write_task_yaml(
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
    potential_kind: str,
    eam_file: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("task_version: 1\n")
        f.write("units: metal\n")
        f.write("box:\n")
        f.write(f"  x: {box:.6f}\n")
        f.write(f"  y: {box:.6f}\n")
        f.write(f"  z: {box:.6f}\n")
        f.write("  pbc: [true, true, true]\n")
        f.write("atoms:\n")
        for idx in range(int(positions.shape[0])):
            rid = idx + 1
            t = int(types[idx])
            m = float(masses[idx])
            rx, ry, rz = positions[idx]
            vx, vy, vz = velocities[idx]
            f.write(f"  - id: {rid}\n")
            f.write(f"    type: {t}\n")
            f.write(f"    mass: {m:.7f}\n")
            f.write(f"    r: [{rx:.6f}, {ry:.6f}, {rz:.6f}]\n")
            f.write(f"    v: [{vx:.6f}, {vy:.6f}, {vz:.6f}]\n")
        pk = str(potential_kind).strip().lower()
        f.write("potential:\n")
        if pk in ("eam/alloy", "eam_alloy"):
            f.write("  kind: eam/alloy\n")
            f.write("  params:\n")
            f.write(f"    file: {eam_file}\n")
            f.write("    elements: [Al, Cu]\n")
        elif pk == "morse":
            f.write("  kind: morse\n")
            f.write("  params:\n")
            f.write("    pair_coeffs:\n")
            f.write("      \"1-1\": {D_e: 0.35, a: 1.20, r0: 2.86}\n")
            f.write("      \"1-2\": {D_e: 0.38, a: 1.25, r0: 2.70}\n")
            f.write("      \"2-2\": {D_e: 0.42, a: 1.30, r0: 2.56}\n")
        else:
            raise ValueError(f"unsupported --potential-kind: {potential_kind!r}")
        f.write(f"cutoff: {cutoff:.6f}\n")
        f.write(f"dt: {dt:.6f}\n")
        f.write(f"steps: {int(steps)}\n")
        f.write("ensemble:\n")
        f.write("  kind: nvt\n")
        f.write("  thermostat:\n")
        f.write("    kind: berendsen\n")
        f.write("    params:\n")
        f.write("      t_target: 600.0\n")
        f.write("      tau: 0.100\n")
        f.write("      every: 1\n")
        f.write("      max_scale_step: 0.10\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Al/Cu crack-healing task with exact atom count")
    p.add_argument(
        "--out",
        default="examples/interop/task_al_cu_crack_100k_nvt.yaml",
        help="Output task YAML path",
    )
    p.add_argument("--target-atoms", type=int, default=100000, help="Exact total atom count")
    p.add_argument("--box", type=float, default=121.5, help="Cubic box size (Angstrom)")
    p.add_argument("--a-al", type=float, default=4.05, help="FCC lattice constant for Al")
    p.add_argument("--a-cu", type=float, default=3.615, help="FCC lattice constant for Cu")
    p.add_argument("--dt", type=float, default=0.001, help="Time step")
    p.add_argument("--steps", type=int, default=2000, help="MD steps")
    p.add_argument("--cutoff", type=float, default=6.5, help="Pair cutoff")
    p.add_argument(
        "--potential-kind",
        default="eam/alloy",
        choices=["eam/alloy", "eam_alloy", "morse"],
        help="Potential model written to task potential block",
    )
    p.add_argument(
        "--eam-file",
        default="examples/potentials/eam_alloy/AlCu.eam.alloy",
        help="EAM/alloy setfl file used when --potential-kind=eam/alloy",
    )
    p.add_argument("--seed", type=int, default=42, help="Velocity RNG seed")
    p.add_argument("--velocity-std", type=float, default=0.020, help="Initial velocity sigma")
    args = p.parse_args()

    box = float(args.box)
    target = int(args.target_atoms)
    if box <= 0.0:
        raise ValueError("box must be positive")
    if target < 1000:
        raise ValueError("target-atoms is too small for this generator")

    # 1) Al matrix.
    al_all = _fcc_points(box=box, a=float(args.a_al))

    crack_lo = np.asarray([0.15 * box, 0.36 * box, 0.02 * box], dtype=float)
    crack_hi = np.asarray([0.85 * box, 0.64 * box, 0.56 * box], dtype=float)
    keep_al = ~_in_box(al_all, crack_lo, crack_hi)
    al_pos = al_all[keep_al]

    n_al = int(al_pos.shape[0])
    n_cu_need = int(target - n_al)
    if n_cu_need <= 0:
        raise RuntimeError(
            f"crack removed too few Al atoms: Al_after_crack={n_al} >= target={target}; "
            "increase crack volume or target-atoms"
        )

    # 2) Cu nanoparticle candidates in crack mouth.
    cu_all = _fcc_points(box=box, a=float(args.a_cu))
    in_crack_cu = _in_box(cu_all, crack_lo, crack_hi)
    cu_pool = cu_all[in_crack_cu]
    if cu_pool.shape[0] == 0:
        raise RuntimeError("no Cu candidate points in crack volume")

    centers = np.asarray(
        [
            [0.45 * box, 0.50 * box, 0.52 * box],
            [0.50 * box, 0.50 * box, 0.52 * box],
            [0.55 * box, 0.50 * box, 0.52 * box],
        ],
        dtype=float,
    )
    d2 = np.sum((cu_pool[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    d2_min = np.min(d2, axis=1)
    order = np.argsort(d2_min)
    if order.size < n_cu_need:
        raise RuntimeError(
            f"not enough Cu candidate points: need={n_cu_need}, available={int(order.size)}; "
            "increase crack volume or reduce target-atoms"
        )
    cu_pos = cu_pool[order[:n_cu_need]]

    # 3) Compose final system with exact target count.
    pos = np.vstack([al_pos, cu_pos])
    types = np.concatenate(
        [
            np.ones((al_pos.shape[0],), dtype=np.int32),
            np.full((cu_pos.shape[0],), 2, dtype=np.int32),
        ]
    )
    if int(pos.shape[0]) != target:
        raise RuntimeError(f"internal mismatch: got {int(pos.shape[0])}, expected {target}")

    masses = np.where(types == 1, 26.9815385, 63.5460).astype(float)

    rng = np.random.default_rng(int(args.seed))
    vel = rng.normal(0.0, float(args.velocity_std), size=(target, 3))
    vel -= np.mean(vel, axis=0, keepdims=True)

    out_path = Path(args.out)
    _write_task_yaml(
        out_path=out_path,
        box=box,
        dt=float(args.dt),
        steps=int(args.steps),
        cutoff=float(args.cutoff),
        positions=pos,
        types=types,
        masses=masses,
        velocities=vel,
        potential_kind=str(args.potential_kind),
        eam_file=str(args.eam_file),
    )

    print(f"[ok] wrote {out_path}")
    print(f"[summary] total={target}  Al={int(np.sum(types==1))}  Cu={int(np.sum(types==2))}")
    print(
        f"[crack] lo=({crack_lo[0]:.3f},{crack_lo[1]:.3f},{crack_lo[2]:.3f}) "
        f"hi=({crack_hi[0]:.3f},{crack_hi[1]:.3f},{crack_hi[2]:.3f})"
    )


if __name__ == "__main__":
    main()
