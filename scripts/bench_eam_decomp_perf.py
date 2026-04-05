#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from tdmd.backend import resolve_backend
from tdmd.potentials import make_potential
from tdmd.td_local import run_td_local


def _fcc_points(*, n_cells: int, lattice_a: float) -> np.ndarray:
    basis = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        dtype=float,
    )
    pts: list[np.ndarray] = []
    for ix in range(int(n_cells)):
        ox = float(ix) * float(lattice_a)
        for iy in range(int(n_cells)):
            oy = float(iy) * float(lattice_a)
            for iz in range(int(n_cells)):
                oz = float(iz) * float(lattice_a)
                pts.append((basis * float(lattice_a)) + np.asarray([ox, oy, oz], dtype=float))
    if not pts:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(pts)


def _build_alloy_state(
    *,
    n_atoms: int,
    lattice_a: float,
    jitter: float,
    seed: int,
    velocity_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if int(n_atoms) <= 0:
        raise ValueError("n_atoms must be positive")
    n_cells = int(math.ceil((float(n_atoms) / 4.0) ** (1.0 / 3.0)))
    n_cells = max(1, n_cells)
    box = float(n_cells) * float(lattice_a)
    pts = _fcc_points(n_cells=n_cells, lattice_a=float(lattice_a))
    if pts.shape[0] < int(n_atoms):
        raise RuntimeError(
            f"not enough FCC points for n_atoms={n_atoms}: have {int(pts.shape[0])}, need {int(n_atoms)}"
        )
    r = np.asarray(pts[: int(n_atoms)], dtype=np.float64).copy()
    rng = np.random.default_rng(int(seed))
    if float(jitter) > 0.0:
        r += rng.normal(0.0, float(jitter), size=r.shape)
        r[:] = np.mod(r, box)

    atom_types = np.ones((int(n_atoms),), dtype=np.int32)
    mix_idx = rng.permutation(int(n_atoms))[: int(n_atoms) // 2]
    atom_types[mix_idx] = 2
    masses = np.where(atom_types == 1, 26.9815385, 63.5460).astype(np.float64)

    v = rng.normal(0.0, float(velocity_std), size=(int(n_atoms), 3)).astype(np.float64)
    v -= np.mean(v, axis=0, keepdims=True)
    return r, v, masses, atom_types, box


def _run_case(
    *,
    label: str,
    requested_device: str,
    decomposition: str,
    r0: np.ndarray,
    v0: np.ndarray,
    masses: np.ndarray,
    atom_types: np.ndarray,
    box: float,
    potential,
    dt: float,
    cutoff: float,
    steps: int,
    repeats: int,
    warmup: int,
    cell_size: float,
    zones_total: int,
    zone_cells_w: int,
    zone_cells_s: int,
    zones_nx: int,
    zones_ny: int,
    zones_nz: int,
) -> dict[str, object]:
    backend = resolve_backend(str(requested_device))
    effective_device = str(getattr(backend, "device", "cpu"))
    fallback_from_cuda = bool(str(requested_device) == "cuda" and effective_device != "cuda")
    times: list[float] = []
    error = ""

    def _run_once() -> float:
        r = np.asarray(r0, dtype=np.float64).copy()
        v = np.asarray(v0, dtype=np.float64).copy()
        t0 = time.perf_counter()
        run_td_local(
            r,
            v,
            masses,
            float(box),
            potential,
            float(dt),
            float(cutoff),
            int(steps),
            observer_every=0,
            atom_types=atom_types,
            cell_size=float(cell_size),
            zones_total=int(zones_total),
            zone_cells_w=int(zone_cells_w),
            zone_cells_s=int(zone_cells_s),
            traversal="forward",
            buffer_k=1.2,
            skin_from_buffer=True,
            use_verlet=True,
            verlet_k_steps=max(4, int(steps)),
            decomposition=str(decomposition),
            sync_mode=False,
            zones_nx=int(zones_nx),
            zones_ny=int(zones_ny),
            zones_nz=int(zones_nz),
            strict_min_zone_width=True,
            ensemble_kind="nve",
            device=str(requested_device),
        )
        return float(time.perf_counter() - t0)

    try:
        for _ in range(max(0, int(warmup))):
            _run_once()
        for _ in range(max(1, int(repeats))):
            times.append(_run_once())
    except Exception as exc:
        error = str(exc)

    ok = bool(times) and not error
    median_sec = float(np.median(times)) if times else 0.0
    mean_sec = float(np.mean(times)) if times else 0.0
    min_sec = float(np.min(times)) if times else 0.0
    max_sec = float(np.max(times)) if times else 0.0
    return {
        "case": str(label),
        "ok": bool(ok),
        "requested_device": str(requested_device),
        "effective_device": effective_device,
        "fallback_from_cuda": int(fallback_from_cuda),
        "decomposition_kind": ("time" if str(decomposition) == "1d" else "space"),
        "decomposition": str(decomposition),
        "repeats": int(repeats),
        "warmup": int(warmup),
        "elapsed_sec_median": median_sec,
        "elapsed_sec_mean": mean_sec,
        "elapsed_sec_min": min_sec,
        "elapsed_sec_max": max_sec,
        "steps_per_sec_median": (float(steps) / median_sec) if median_sec > 0.0 else 0.0,
        "error": error,
    }


def _speedup(numer: dict[str, object], denom: dict[str, object]) -> float | None:
    t_num = float(numer.get("elapsed_sec_median", 0.0) or 0.0)
    t_den = float(denom.get("elapsed_sec_median", 0.0) or 0.0)
    if t_num <= 0.0 or t_den <= 0.0:
        return None
    return float(t_num / t_den)


def _fmt_time(v: float) -> str:
    return f"{float(v):.6f}s" if float(v) > 0.0 else "n/a"


def _fmt_steps_per_sec(v: float) -> str:
    return f"{float(v):.3f}" if float(v) > 0.0 else "n/a"


def _fmt_speedup(v: float | None) -> str:
    return f"{float(v):.3f}x" if v is not None else "n/a"


def _parse_cases(cases_arg: str) -> list[str]:
    full_order = ["space_cpu", "space_gpu", "time_cpu", "time_gpu"]
    seen: set[str] = set()
    cases: list[str] = []
    for part in str(cases_arg).split(","):
        case = str(part).strip()
        if not case:
            continue
        if case not in full_order:
            raise ValueError(
                f"unsupported case {case!r}; supported cases are: {', '.join(full_order)}"
            )
        if case not in seen:
            cases.append(case)
            seen.add(case)
    if not cases:
        raise ValueError("at least one benchmark case must be selected")
    return cases


def _build_markdown_report(
    *,
    rows_by_case: dict[str, dict[str, object]],
    case_order: list[str],
    n_atoms: int,
    steps: int,
    repeats: int,
    warmup: int,
) -> str:
    col_order = [case for case in case_order if case in rows_by_case]

    def _metric_row(metric: str, formatter) -> str:
        return "| " + metric + " | " + " | ".join(formatter(case) for case in col_order) + " |"

    def _gpu_speedup_cell(case: str) -> str:
        if case == "space_gpu" and "space_cpu" in rows_by_case:
            row = rows_by_case["space_gpu"]
            return (
                _fmt_speedup(_speedup(rows_by_case["space_cpu"], row))
                if str(row.get("effective_device", "cpu")) == "cuda"
                else "n/a"
            )
        if case == "time_gpu" and "time_cpu" in rows_by_case:
            row = rows_by_case["time_gpu"]
            return (
                _fmt_speedup(_speedup(rows_by_case["time_cpu"], row))
                if str(row.get("effective_device", "cpu")) == "cuda"
                else "n/a"
            )
        return "-"

    def _td_speedup_cell(case: str) -> str:
        if case == "time_cpu" and "space_cpu" in rows_by_case:
            return _fmt_speedup(_speedup(rows_by_case["space_cpu"], rows_by_case["time_cpu"]))
        if case == "time_gpu" and "space_gpu" in rows_by_case:
            space_gpu = rows_by_case["space_gpu"]
            time_gpu = rows_by_case["time_gpu"]
            if str(space_gpu.get("effective_device", "cpu")) == str(
                time_gpu.get("effective_device", "cpu")
            ):
                return _fmt_speedup(_speedup(space_gpu, time_gpu))
            return "n/a"
        return "-"

    lines = [
        "# EAM Alloy Decomposition Benchmark",
        "",
        f"- n_atoms: `{int(n_atoms)}`",
        f"- steps: `{int(steps)}`",
        f"- repeats: `{int(repeats)}`",
        f"- warmup: `{int(warmup)}`",
        f"- selected_cases: `{','.join(col_order)}`",
        "- interpretation: `time=* td_local(decomposition=1d)` and `space=* td_local(decomposition=3d)`",
        "",
        "| metric | " + " | ".join(col_order) + " |",
        "|" + "---|" * (len(col_order) + 1),
        _metric_row(
            "effective_device",
            lambda case: f"`{str(rows_by_case[case].get('effective_device', 'cpu'))}`",
        ),
        _metric_row(
            "fallback_from_cuda",
            lambda case: str(int(rows_by_case[case].get("fallback_from_cuda", 0))),
        ),
        _metric_row(
            "median_sec",
            lambda case: _fmt_time(float(rows_by_case[case].get("elapsed_sec_median", 0.0) or 0.0)),
        ),
        _metric_row(
            "mean_sec",
            lambda case: _fmt_time(float(rows_by_case[case].get("elapsed_sec_mean", 0.0) or 0.0)),
        ),
        _metric_row(
            "steps_per_sec",
            lambda case: _fmt_steps_per_sec(
                float(rows_by_case[case].get("steps_per_sec_median", 0.0) or 0.0)
            ),
        ),
    ]
    if ("space_cpu" in rows_by_case and "space_gpu" in rows_by_case) or (
        "time_cpu" in rows_by_case and "time_gpu" in rows_by_case
    ):
        lines.append(_metric_row("gpu_speedup_vs_cpu", _gpu_speedup_cell))
    if any(case in rows_by_case for case in ("time_cpu", "time_gpu")):
        lines.append(_metric_row("td_speedup_vs_space", _td_speedup_cell))
    lines.append("")
    lines.append("## Case Summary")
    lines.append("")
    for case in col_order:
        row = rows_by_case[case]
        lines.append(
            f"- `{case}` ok={bool(row.get('ok', False))} "
            f"requested=`{row.get('requested_device', '')}` "
            f"effective=`{row.get('effective_device', '')}` "
            f"median={_fmt_time(float(row.get('elapsed_sec_median', 0.0) or 0.0))} "
            f"error=`{row.get('error', '')}`"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark EAM/alloy TD-vs-space decomposition on CPU/GPU")
    ap.add_argument("--out", default="results/eam_decomp_perf.csv")
    ap.add_argument("--md", default="results/eam_decomp_perf.md")
    ap.add_argument("--json", default="results/eam_decomp_perf.summary.json")
    ap.add_argument("--n-atoms", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lattice-a", type=float, default=4.05)
    ap.add_argument("--jitter", type=float, default=0.02)
    ap.add_argument("--velocity-std", type=float, default=0.01)
    ap.add_argument("--cutoff", type=float, default=6.5)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--cell-size", type=float, default=2.0)
    ap.add_argument("--zones-total", type=int, default=2)
    ap.add_argument("--zone-cells-w", type=int, default=1)
    ap.add_argument("--zone-cells-s", type=int, default=2)
    ap.add_argument("--zones-nx", type=int, default=2)
    ap.add_argument("--zones-ny", type=int, default=1)
    ap.add_argument("--zones-nz", type=int, default=1)
    ap.add_argument(
        "--cases",
        default="space_cpu,space_gpu,time_cpu,time_gpu",
        help="comma-separated subset of benchmark cases: space_cpu,space_gpu,time_cpu,time_gpu",
    )
    ap.add_argument(
        "--eam-file",
        default="examples/potentials/eam_alloy/AlCu.eam.alloy",
        help="EAM/alloy setfl file used for the benchmark potential",
    )
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--require-effective-cuda", action="store_true")
    args = ap.parse_args()

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

    case_order = _parse_cases(args.cases)
    case_specs = {
        "space_cpu": ("space_cpu", "cpu", "3d"),
        "space_gpu": ("space_gpu", "cuda", "3d"),
        "time_cpu": ("time_cpu", "cpu", "1d"),
        "time_gpu": ("time_gpu", "cuda", "1d"),
    }
    rows: list[dict[str, object]] = []
    for case in case_order:
        label, requested_device, decomposition = case_specs[case]
        rows.append(
            _run_case(
                label=label,
                requested_device=requested_device,
                decomposition=decomposition,
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
                zones_total=int(args.zones_total),
                zone_cells_w=int(args.zone_cells_w),
                zone_cells_s=int(args.zone_cells_s),
                zones_nx=int(args.zones_nx),
                zones_ny=int(args.zones_ny),
                zones_nz=int(args.zones_nz),
            )
        )

    rows_by_case = {str(r["case"]): r for r in rows}
    report_markdown = _build_markdown_report(
        rows_by_case=rows_by_case,
        case_order=case_order,
        n_atoms=int(args.n_atoms),
        steps=int(args.steps),
        repeats=int(args.repeats),
        warmup=int(args.warmup),
    )

    gpu_cases = [rows_by_case[case] for case in case_order if case.endswith("_gpu")]
    gpu_effective_ok = all(str(r.get("effective_device", "cpu")) == "cuda" for r in gpu_cases)
    ok_all = all(bool(r.get("ok", False)) for r in rows)
    if bool(args.require_effective_cuda):
        ok_all = bool(ok_all and gpu_effective_ok)

    comparisons = {
        "gpu_speedup_space": (
            _speedup(rows_by_case["space_cpu"], rows_by_case["space_gpu"])
            if "space_cpu" in rows_by_case
            and "space_gpu" in rows_by_case
            and str(rows_by_case["space_gpu"].get("effective_device", "cpu")) == "cuda"
            else None
        ),
        "gpu_speedup_time": (
            _speedup(rows_by_case["time_cpu"], rows_by_case["time_gpu"])
            if "time_cpu" in rows_by_case
            and "time_gpu" in rows_by_case
            and str(rows_by_case["time_gpu"].get("effective_device", "cpu")) == "cuda"
            else None
        ),
        "time_speedup_cpu": (
            _speedup(rows_by_case["space_cpu"], rows_by_case["time_cpu"])
            if "space_cpu" in rows_by_case and "time_cpu" in rows_by_case
            else None
        ),
        "time_speedup_gpu": (
            _speedup(rows_by_case["space_gpu"], rows_by_case["time_gpu"])
            if "space_gpu" in rows_by_case
            and "time_gpu" in rows_by_case
            and str(rows_by_case["space_gpu"].get("effective_device", "cpu"))
            == str(rows_by_case["time_gpu"].get("effective_device", "cpu"))
            else None
        ),
    }

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
                "case",
                "decomposition_kind",
                "requested_device",
                "effective_device",
                "fallback_from_cuda",
                "ok",
                "elapsed_sec_median",
                "elapsed_sec_mean",
                "elapsed_sec_min",
                "elapsed_sec_max",
                "steps_per_sec_median",
                "repeats",
                "warmup",
                "error",
            ]
        )
        for row in rows:
            w.writerow(
                [
                    row["case"],
                    row["decomposition_kind"],
                    row["requested_device"],
                    row["effective_device"],
                    row["fallback_from_cuda"],
                    int(bool(row["ok"])),
                    f"{float(row['elapsed_sec_median']):.6f}",
                    f"{float(row['elapsed_sec_mean']):.6f}",
                    f"{float(row['elapsed_sec_min']):.6f}",
                    f"{float(row['elapsed_sec_max']):.6f}",
                    f"{float(row['steps_per_sec_median']):.6f}",
                    int(row["repeats"]),
                    int(row["warmup"]),
                    row["error"],
                ]
            )

    out_md.write_text(report_markdown, encoding="utf-8")

    summary = {
        "n": len(rows),
        "total": len(rows),
        "ok": int(sum(1 for r in rows if bool(r.get("ok", False)))),
        "fail": int(sum(1 for r in rows if not bool(r.get("ok", False)))),
        "ok_all": bool(ok_all),
        "worst": {
            "max_elapsed_sec_median": max(float(r.get("elapsed_sec_median", 0.0) or 0.0) for r in rows),
            "gpu_effective_ok": int(bool(gpu_effective_ok)),
        },
        "rows": rows,
        "by_case": {str(r["case"]): dict(r) for r in rows},
        "comparisons": comparisons,
        "selected_cases": case_order,
        "report_markdown": report_markdown,
        "effective_cuda_required": bool(args.require_effective_cuda),
        "gpu_effective_ok": bool(gpu_effective_ok),
        "artifacts": {
            "csv": str(out_csv),
            "md": str(out_md),
            "json": str(out_json),
        },
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(report_markdown, end="")
    print(f"[eam-decomp-perf] wrote {out_csv}")
    print(f"[eam-decomp-perf] wrote {out_md}")
    print(f"[eam-decomp-perf] wrote {out_json}")

    if bool(args.strict) and not bool(summary["ok_all"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
