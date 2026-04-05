#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from bench_eam_decomp_perf import (
    _build_alloy_state,
    _fmt_speedup,
    _fmt_steps_per_sec,
    _fmt_time,
    _run_case,
    _speedup,
)
from tdmd.potentials import make_potential


def _parse_layouts(layouts_arg: str) -> list[tuple[int, int, int, int]]:
    layouts: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for part in str(layouts_arg).split(","):
        spec = str(part).strip()
        if not spec:
            continue
        total_str, dims_str = spec.split(":", 1)
        total = int(total_str)
        dims = [int(v) for v in dims_str.lower().split("x")]
        if len(dims) != 3:
            raise ValueError(f"invalid layout {spec!r}; expected '<zones_total>:<nx>x<ny>x<nz>'")
        nx, ny, nz = dims
        if min(total, nx, ny, nz) <= 0:
            raise ValueError(f"layout values must be positive: {spec!r}")
        if total != int(nx * ny * nz):
            raise ValueError(
                f"layout total {total} does not match nx*ny*nz={int(nx * ny * nz)} in {spec!r}"
            )
        key = (int(total), int(nx), int(ny), int(nz))
        if key not in seen:
            layouts.append(key)
            seen.add(key)
    if not layouts:
        raise ValueError("at least one layout must be provided")
    return layouts


def _build_markdown_report(
    *,
    rows: list[dict[str, object]],
    n_atoms: int,
    steps: int,
    repeats: int,
    warmup: int,
    layouts_arg: str,
) -> str:
    lines = [
        "# EAM Alloy GPU Zone Sweep",
        "",
        f"- n_atoms: `{int(n_atoms)}`",
        f"- steps: `{int(steps)}`",
        f"- repeats: `{int(repeats)}`",
        f"- warmup: `{int(warmup)}`",
        f"- layouts: `{layouts_arg}`",
        "- interpretation: `space_gpu` uses `decomposition=3d`; `time_gpu` uses `decomposition=1d` with the same total zone count.",
        "",
        "| zones_total | space_layout | space_gpu | time_gpu | td_speedup_vs_space | space_steps_per_sec | time_steps_per_sec |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + str(int(row["zones_total"]))
            + " | "
            + f"`{row['space_layout']}`"
            + " | "
            + _fmt_time(float(row.get("space_gpu_median_sec", 0.0) or 0.0))
            + " | "
            + _fmt_time(float(row.get("time_gpu_median_sec", 0.0) or 0.0))
            + " | "
            + _fmt_speedup(row.get("td_speedup_vs_space"))
            + " | "
            + _fmt_steps_per_sec(float(row.get("space_gpu_steps_per_sec", 0.0) or 0.0))
            + " | "
            + _fmt_steps_per_sec(float(row.get("time_gpu_steps_per_sec", 0.0) or 0.0))
            + " |"
        )

    ok_rows = [row for row in rows if bool(row.get("ok", False))]
    if ok_rows:
        best_td = max(ok_rows, key=lambda row: float(row.get("td_speedup_vs_space", 0.0) or 0.0))
        best_time = min(
            ok_rows,
            key=lambda row: float(row.get("time_gpu_median_sec", float("inf")) or float("inf")),
        )
        lines.extend(
            [
                "",
                "## Best Observed Layouts",
                "",
                f"- best_td_speedup: zones_total=`{int(best_td['zones_total'])}` layout=`{best_td['space_layout']}` td_speedup_vs_space=`{_fmt_speedup(best_td.get('td_speedup_vs_space'))}`",
                f"- fastest_time_gpu: zones_total=`{int(best_time['zones_total'])}` layout=`{best_time['space_layout']}` time_gpu=`{_fmt_time(float(best_time.get('time_gpu_median_sec', 0.0) or 0.0))}`",
            ]
        )

    lines.extend(["", "## Layout Summary", ""])
    for row in rows:
        lines.append(
            f"- zones_total=`{int(row['zones_total'])}` layout=`{row['space_layout']}` "
            f"ok={bool(row.get('ok', False))} "
            f"space_effective=`{row.get('space_gpu_effective_device', '')}` "
            f"time_effective=`{row.get('time_gpu_effective_device', '')}` "
            f"space_error=`{row.get('space_gpu_error', '')}` "
            f"time_error=`{row.get('time_gpu_error', '')}`"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="GPU-only EAM/alloy zone sweep comparing TD (1d) against space decomposition (3d)"
    )
    ap.add_argument("--out", default="results/eam_zone_sweep_gpu.csv")
    ap.add_argument("--md", default="results/eam_zone_sweep_gpu.md")
    ap.add_argument("--json", default="results/eam_zone_sweep_gpu.summary.json")
    ap.add_argument("--n-atoms", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lattice-a", type=float, default=4.05)
    ap.add_argument("--jitter", type=float, default=0.02)
    ap.add_argument("--velocity-std", type=float, default=0.01)
    ap.add_argument("--cutoff", type=float, default=6.5)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--cell-size", type=float, default=2.0)
    ap.add_argument("--zone-cells-w", type=int, default=1)
    ap.add_argument("--zone-cells-s", type=int, default=2)
    ap.add_argument(
        "--layouts",
        default="2:2x1x1,4:2x2x1,6:3x2x1",
        help="comma-separated layout list '<zones_total>:<nx>x<ny>x<nz>'",
    )
    ap.add_argument(
        "--eam-file",
        default="examples/potentials/eam_alloy/AlCu.eam.alloy",
        help="EAM/alloy setfl file used for the benchmark potential",
    )
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--require-effective-cuda", action="store_true")
    args = ap.parse_args()

    layouts = _parse_layouts(args.layouts)
    r0, v0, masses, atom_types, box = _build_alloy_state(
        n_atoms=int(args.n_atoms),
        lattice_a=float(args.lattice_a),
        jitter=float(args.jitter),
        seed=int(args.seed),
        velocity_std=float(args.velocity_std),
    )
    potential = make_potential(
        "eam/alloy",
        {"file": str(args.eam_file), "elements": ["Al", "Cu"]},
    )

    rows: list[dict[str, object]] = []
    for zones_total, zones_nx, zones_ny, zones_nz in layouts:
        space_case = _run_case(
            label=f"space_gpu_z{zones_total}",
            requested_device="cuda",
            decomposition="3d",
            r0=r0,
            v0=v0,
            masses=masses,
            atom_types=atom_types,
            box=box,
            potential=potential,
            dt=float(args.dt),
            cutoff=float(args.cutoff),
            steps=int(args.steps),
            repeats=int(args.repeats),
            warmup=int(args.warmup),
            cell_size=float(args.cell_size),
            zones_total=int(zones_total),
            zone_cells_w=int(args.zone_cells_w),
            zone_cells_s=int(args.zone_cells_s),
            zones_nx=int(zones_nx),
            zones_ny=int(zones_ny),
            zones_nz=int(zones_nz),
        )
        time_case = _run_case(
            label=f"time_gpu_z{zones_total}",
            requested_device="cuda",
            decomposition="1d",
            r0=r0,
            v0=v0,
            masses=masses,
            atom_types=atom_types,
            box=box,
            potential=potential,
            dt=float(args.dt),
            cutoff=float(args.cutoff),
            steps=int(args.steps),
            repeats=int(args.repeats),
            warmup=int(args.warmup),
            cell_size=float(args.cell_size),
            zones_total=int(zones_total),
            zone_cells_w=int(args.zone_cells_w),
            zone_cells_s=int(args.zone_cells_s),
            zones_nx=int(zones_nx),
            zones_ny=int(zones_ny),
            zones_nz=int(zones_nz),
        )
        gpu_effective_ok = (
            str(space_case.get("effective_device", "cpu")) == "cuda"
            and str(time_case.get("effective_device", "cpu")) == "cuda"
        )
        row_ok = bool(space_case.get("ok", False) and time_case.get("ok", False))
        if bool(args.require_effective_cuda):
            row_ok = bool(row_ok and gpu_effective_ok)
        rows.append(
            {
                "zones_total": int(zones_total),
                "space_layout": f"{int(zones_nx)}x{int(zones_ny)}x{int(zones_nz)}",
                "ok": bool(row_ok),
                "gpu_effective_ok": bool(gpu_effective_ok),
                "space_gpu_median_sec": float(space_case.get("elapsed_sec_median", 0.0) or 0.0),
                "time_gpu_median_sec": float(time_case.get("elapsed_sec_median", 0.0) or 0.0),
                "space_gpu_steps_per_sec": float(
                    space_case.get("steps_per_sec_median", 0.0) or 0.0
                ),
                "time_gpu_steps_per_sec": float(
                    time_case.get("steps_per_sec_median", 0.0) or 0.0
                ),
                "td_speedup_vs_space": _speedup(space_case, time_case),
                "space_gpu_effective_device": str(space_case.get("effective_device", "")),
                "time_gpu_effective_device": str(time_case.get("effective_device", "")),
                "space_gpu_fallback_from_cuda": int(space_case.get("fallback_from_cuda", 0) or 0),
                "time_gpu_fallback_from_cuda": int(time_case.get("fallback_from_cuda", 0) or 0),
                "space_gpu_error": str(space_case.get("error", "")),
                "time_gpu_error": str(time_case.get("error", "")),
                "space_gpu_case": dict(space_case),
                "time_gpu_case": dict(time_case),
            }
        )

    report_markdown = _build_markdown_report(
        rows=rows,
        n_atoms=int(args.n_atoms),
        steps=int(args.steps),
        repeats=int(args.repeats),
        warmup=int(args.warmup),
        layouts_arg=str(args.layouts),
    )

    ok_all = all(bool(row.get("ok", False)) for row in rows)
    out_csv = Path(args.out)
    out_md = Path(args.md)
    out_json = Path(args.json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "zones_total",
                "space_layout",
                "ok",
                "gpu_effective_ok",
                "space_gpu_effective_device",
                "time_gpu_effective_device",
                "space_gpu_median_sec",
                "time_gpu_median_sec",
                "space_gpu_steps_per_sec",
                "time_gpu_steps_per_sec",
                "td_speedup_vs_space",
                "space_gpu_error",
                "time_gpu_error",
            ]
        )
        for row in rows:
            w.writerow(
                [
                    int(row["zones_total"]),
                    row["space_layout"],
                    int(bool(row["ok"])),
                    int(bool(row["gpu_effective_ok"])),
                    row["space_gpu_effective_device"],
                    row["time_gpu_effective_device"],
                    f"{float(row['space_gpu_median_sec']):.6f}",
                    f"{float(row['time_gpu_median_sec']):.6f}",
                    f"{float(row['space_gpu_steps_per_sec']):.6f}",
                    f"{float(row['time_gpu_steps_per_sec']):.6f}",
                    (
                        f"{float(row['td_speedup_vs_space']):.6f}"
                        if row["td_speedup_vs_space"] is not None
                        else ""
                    ),
                    row["space_gpu_error"],
                    row["time_gpu_error"],
                ]
            )

    out_md.write_text(report_markdown, encoding="utf-8")

    ok_rows = [row for row in rows if bool(row.get("ok", False))]
    best_layouts = {
        "best_td_speedup": (
            {
                "zones_total": int(
                    max(ok_rows, key=lambda row: float(row.get("td_speedup_vs_space", 0.0) or 0.0))[
                        "zones_total"
                    ]
                ),
                "space_layout": str(
                    max(ok_rows, key=lambda row: float(row.get("td_speedup_vs_space", 0.0) or 0.0))[
                        "space_layout"
                    ]
                ),
                "td_speedup_vs_space": float(
                    max(ok_rows, key=lambda row: float(row.get("td_speedup_vs_space", 0.0) or 0.0)).get(
                        "td_speedup_vs_space", 0.0
                    )
                    or 0.0
                ),
            }
            if ok_rows
            else None
        ),
        "fastest_time_gpu": (
            {
                "zones_total": int(
                    min(ok_rows, key=lambda row: float(row.get("time_gpu_median_sec", float("inf")) or float("inf")))[
                        "zones_total"
                    ]
                ),
                "space_layout": str(
                    min(ok_rows, key=lambda row: float(row.get("time_gpu_median_sec", float("inf")) or float("inf")))[
                        "space_layout"
                    ]
                ),
                "time_gpu_median_sec": float(
                    min(ok_rows, key=lambda row: float(row.get("time_gpu_median_sec", float("inf")) or float("inf"))).get(
                        "time_gpu_median_sec", 0.0
                    )
                    or 0.0
                ),
            }
            if ok_rows
            else None
        ),
    }
    summary = {
        "n": len(rows),
        "total": len(rows),
        "ok": int(sum(1 for row in rows if bool(row.get("ok", False)))),
        "fail": int(sum(1 for row in rows if not bool(row.get("ok", False)))),
        "ok_all": bool(ok_all),
        "selected_layouts": [f"{z}:{nx}x{ny}x{nz}" for z, nx, ny, nz in layouts],
        "worst": {
            "max_space_gpu_median_sec": max(
                (float(row.get("space_gpu_median_sec", 0.0) or 0.0) for row in rows),
                default=0.0,
            ),
            "max_time_gpu_median_sec": max(
                (float(row.get("time_gpu_median_sec", 0.0) or 0.0) for row in rows),
                default=0.0,
            ),
            "min_td_speedup_vs_space": min(
                (
                    float(row.get("td_speedup_vs_space", 0.0) or 0.0)
                    for row in rows
                    if row.get("td_speedup_vs_space") is not None
                ),
                default=0.0,
            ),
            "gpu_effective_ok_all": int(
                all(bool(row.get("gpu_effective_ok", False)) for row in rows)
            ),
        },
        "rows": rows,
        "by_layout": {str(row["space_layout"]): dict(row) for row in rows},
        "best_layouts": best_layouts,
        "report_markdown": report_markdown,
        "effective_cuda_required": bool(args.require_effective_cuda),
        "artifacts": {
            "csv": str(out_csv),
            "md": str(out_md),
            "json": str(out_json),
        },
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(report_markdown, end="")
    print(f"[eam-zone-sweep-gpu] wrote {out_csv}")
    print(f"[eam-zone-sweep-gpu] wrote {out_md}")
    print(f"[eam-zone-sweep-gpu] wrote {out_json}")
    if bool(args.strict) and not bool(summary["ok_all"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
