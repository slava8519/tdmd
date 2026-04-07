#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bench_eam_decomp_perf import _build_alloy_state
from tdmd.config import load_config
from tdmd.potentials import make_potential
from tdmd.state import init_positions, init_velocities
from tdmd.wavefront_reference import (
    WAVEFRONT_REFERENCE_CONTRACT_VERSION,
    prove_wavefront_1d_reference_equivalence,
)


def _run_morse_cfg_case(
    *,
    config_path: str,
    steps: int,
    zones_total: int,
    cell_size: float,
    zone_cells_w: int,
    zone_cells_s: int,
    atol: float,
    require_multi_zone_wave: bool,
) -> dict[str, object]:
    cfg = load_config(str(config_path))
    potential = make_potential(cfg.potential.kind, cfg.potential.params)
    r0 = init_positions(int(cfg.system.n_atoms), float(cfg.system.box), int(cfg.system.seed))
    v0 = init_velocities(
        int(cfg.system.n_atoms),
        float(cfg.system.temperature),
        float(cfg.system.mass),
        int(cfg.system.seed),
    )
    summary = prove_wavefront_1d_reference_equivalence(
        r0=r0,
        v0=v0,
        mass=float(cfg.system.mass),
        box=float(cfg.system.box),
        potential=potential,
        dt=float(cfg.run.dt),
        cutoff=float(cfg.run.cutoff),
        n_steps=int(steps),
        atom_types=np.ones((int(cfg.system.n_atoms),), dtype=np.int32),
        cell_size=float(cell_size),
        zones_total=int(zones_total),
        zone_cells_w=int(zone_cells_w),
        zone_cells_s=int(zone_cells_s),
        traversal=str(cfg.td.traversal),
        atol=float(atol),
        require_multi_zone_wave=bool(require_multi_zone_wave),
    )
    summary["case"] = "morse_cfg_pair"
    summary["source"] = {"config": str(config_path)}
    return summary


def _run_eam_alloy_dense_case(
    *,
    n_atoms: int,
    lattice_a: float,
    jitter: float,
    seed: int,
    velocity_std: float,
    dt: float,
    cutoff: float,
    steps: int,
    zones_total: int,
    cell_size: float,
    zone_cells_w: int,
    zone_cells_s: int,
    eam_file: str,
    atol: float,
    require_multi_zone_wave: bool,
) -> dict[str, object]:
    r0, v0, masses, atom_types, box = _build_alloy_state(
        n_atoms=int(n_atoms),
        lattice_a=float(lattice_a),
        jitter=float(jitter),
        seed=int(seed),
        velocity_std=float(velocity_std),
    )
    potential = make_potential(
        "eam/alloy",
        {
            "file": str(eam_file),
            "elements": ["Al", "Cu"],
        },
    )
    summary = prove_wavefront_1d_reference_equivalence(
        r0=r0,
        v0=v0,
        mass=masses,
        box=float(box),
        potential=potential,
        dt=float(dt),
        cutoff=float(cutoff),
        n_steps=int(steps),
        atom_types=atom_types,
        cell_size=float(cell_size),
        zones_total=int(zones_total),
        zone_cells_w=int(zone_cells_w),
        zone_cells_s=int(zone_cells_s),
        traversal="forward",
        atol=float(atol),
        require_multi_zone_wave=bool(require_multi_zone_wave),
    )
    summary["case"] = "eam_alloy_dense"
    summary["source"] = {
        "eam_file": str(eam_file),
        "n_atoms": int(n_atoms),
        "lattice_a": float(lattice_a),
        "seed": int(seed),
    }
    return summary


def _build_report(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Wavefront Reference Equivalence",
        "",
        f"- contract_version: `{WAVEFRONT_REFERENCE_CONTRACT_VERSION}`",
        "",
        "| case | ok | many_body | max_force_abs | max_position_abs | max_velocity_abs | max_wave_size | multi_zone_wave_seen | reference_force_kind |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + str(row.get("case", ""))
            + " | "
            + str(int(bool(row.get("ok_all", False))))
            + " | "
            + str(int(bool(row.get("many_body", False))))
            + " | "
            + f"{float(row.get('max_force_max_abs', 0.0) or 0.0):.3e}"
            + " | "
            + f"{float(row.get('max_position_max_abs', 0.0) or 0.0):.3e}"
            + " | "
            + f"{float(row.get('max_velocity_max_abs', 0.0) or 0.0):.3e}"
            + " | "
            + str(int(row.get("max_wave_size", 0) or 0))
            + " | "
            + str(int(bool(row.get("multi_zone_wave_seen", False))))
            + " | "
            + str(row.get("reference_force_kind", ""))
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `ok=1` means the shadow wave-batch trajectory stayed within tolerance of the current CPU/reference sequential slab trajectory.",
            "- `max_wave_size > 1` confirms that the case exercised at least one formally admissible multi-zone slab wave.",
            "- The harness is verification-only; it does not change `td_local` runtime order.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="CPU/reference wave-batch equivalence harness")
    ap.add_argument("--out", default="results/wavefront_reference_equivalence.csv")
    ap.add_argument("--md", default="results/wavefront_reference_equivalence.md")
    ap.add_argument("--json", default="results/wavefront_reference_equivalence.summary.json")
    ap.add_argument(
        "--morse-config",
        default="examples/td_1d_morse.yaml",
    )
    ap.add_argument("--morse-steps", type=int, default=4)
    ap.add_argument("--morse-zones-total", type=int, default=8)
    ap.add_argument("--morse-cell-size", type=float, default=4.0)
    ap.add_argument("--morse-zone-cells-w", type=int, default=1)
    ap.add_argument("--morse-zone-cells-s", type=int, default=2)
    ap.add_argument("--eam-alloy-n-atoms", type=int, default=1000)
    ap.add_argument("--eam-alloy-lattice-a", type=float, default=4.05)
    ap.add_argument("--eam-alloy-jitter", type=float, default=0.02)
    ap.add_argument("--eam-alloy-seed", type=int, default=42)
    ap.add_argument("--eam-alloy-velocity-std", type=float, default=0.01)
    ap.add_argument("--eam-alloy-dt", type=float, default=0.001)
    ap.add_argument("--eam-alloy-cutoff", type=float, default=6.5)
    ap.add_argument("--eam-alloy-steps", type=int, default=2)
    ap.add_argument("--eam-alloy-zones-total", type=int, default=4)
    ap.add_argument("--eam-alloy-cell-size", type=float, default=3.5)
    ap.add_argument("--eam-alloy-zone-cells-w", type=int, default=1)
    ap.add_argument("--eam-alloy-zone-cells-s", type=int, default=1)
    ap.add_argument(
        "--eam-alloy-file",
        default="examples/potentials/eam_alloy/AlCu.eam.alloy",
    )
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--allow-no-multi-zone-wave", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    out_csv = Path(args.out)
    out_md = Path(args.md)
    out_json = Path(args.json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    require_multi_zone_wave = not bool(args.allow_no_multi_zone_wave)
    rows = [
        _run_morse_cfg_case(
            config_path=str(args.morse_config),
            steps=int(args.morse_steps),
            zones_total=int(args.morse_zones_total),
            cell_size=float(args.morse_cell_size),
            zone_cells_w=int(args.morse_zone_cells_w),
            zone_cells_s=int(args.morse_zone_cells_s),
            atol=float(args.atol),
            require_multi_zone_wave=bool(require_multi_zone_wave),
        ),
        _run_eam_alloy_dense_case(
            n_atoms=int(args.eam_alloy_n_atoms),
            lattice_a=float(args.eam_alloy_lattice_a),
            jitter=float(args.eam_alloy_jitter),
            seed=int(args.eam_alloy_seed),
            velocity_std=float(args.eam_alloy_velocity_std),
            dt=float(args.eam_alloy_dt),
            cutoff=float(args.eam_alloy_cutoff),
            steps=int(args.eam_alloy_steps),
            zones_total=int(args.eam_alloy_zones_total),
            cell_size=float(args.eam_alloy_cell_size),
            zone_cells_w=int(args.eam_alloy_zone_cells_w),
            zone_cells_s=int(args.eam_alloy_zone_cells_s),
            eam_file=str(args.eam_alloy_file),
            atol=float(args.atol),
            require_multi_zone_wave=bool(require_multi_zone_wave),
        ),
    ]

    report_markdown = _build_report(rows)
    ok_all = all(bool(row.get("ok_all", False)) for row in rows)
    summary = {
        "contract_version": WAVEFRONT_REFERENCE_CONTRACT_VERSION,
        "n": int(len(rows)),
        "total": int(len(rows)),
        "ok": int(sum(1 for row in rows if bool(row.get("ok_all", False)))),
        "fail": int(sum(1 for row in rows if not bool(row.get("ok_all", False)))),
        "ok_all": bool(ok_all),
        "rows": rows,
        "by_case": {str(row.get("case", "")): dict(row) for row in rows},
        "report_markdown": report_markdown,
        "artifacts": {
            "csv": str(out_csv),
            "md": str(out_md),
            "json": str(out_json),
        },
    }

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "case",
                "ok",
                "many_body",
                "max_force_max_abs",
                "max_position_max_abs",
                "max_velocity_max_abs",
                "max_wave_size",
                "multi_zone_wave_seen",
                "reference_force_kind",
                "shadow_force_kind",
            ]
        )
        for row in rows:
            w.writerow(
                [
                    str(row.get("case", "")),
                    int(bool(row.get("ok_all", False))),
                    int(bool(row.get("many_body", False))),
                    float(row.get("max_force_max_abs", 0.0) or 0.0),
                    float(row.get("max_position_max_abs", 0.0) or 0.0),
                    float(row.get("max_velocity_max_abs", 0.0) or 0.0),
                    int(row.get("max_wave_size", 0) or 0),
                    int(bool(row.get("multi_zone_wave_seen", False))),
                    str(row.get("reference_force_kind", "")),
                    str(row.get("shadow_force_kind", "")),
                ]
            )

    out_md.write_text(report_markdown, encoding="utf-8")
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(report_markdown, end="")
    print(f"[wavefront-reference] wrote {out_csv}")
    print(f"[wavefront-reference] wrote {out_md}")
    print(f"[wavefront-reference] wrote {out_json}")
    if bool(args.strict) and not bool(summary["ok_all"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
