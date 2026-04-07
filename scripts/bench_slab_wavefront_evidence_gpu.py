#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bench_eam_decomp_perf import _build_alloy_state
from generate_al_crack_task import build_al_crack_state, write_al_crack_task_yaml


def _parse_int_list(spec: str) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for part in str(spec).split(","):
        txt = str(part).strip()
        if not txt:
            continue
        value = int(txt)
        if value <= 0:
            raise ValueError(f"value must be positive: {txt!r}")
        if value not in seen:
            out.append(value)
            seen.add(value)
    if not out:
        raise ValueError("at least one integer value must be provided")
    return out


def _factorizations(total: int) -> list[tuple[int, int, int]]:
    out: set[tuple[int, int, int]] = set()
    for nx in range(1, int(total) + 1):
        if int(total) % nx != 0:
            continue
        rem = int(total) // nx
        for ny in range(1, rem + 1):
            if rem % ny != 0:
                continue
            nz = rem // ny
            out.add(tuple(sorted((int(nx), int(ny), int(nz)), reverse=True)))
    return sorted(out)


def _best_space_layout(total: int, box: float, cutoff: float) -> tuple[int, int, int]:
    best: tuple[float, float, tuple[int, int, int]] | None = None
    for nx, ny, nz in _factorizations(int(total)):
        widths = [float(box) / float(nx), float(box) / float(ny), float(box) / float(nz)]
        min_width = min(widths)
        max_width = max(widths)
        if min_width < float(cutoff):
            continue
        score = (min_width / max_width if max_width > 0.0 else 0.0, min_width)
        if best is None or score > (best[0], best[1]):
            best = (score[0], score[1], (int(nx), int(ny), int(nz)))
    if best is None:
        raise ValueError(
            f"no valid 3D layout for zones_total={int(total)} and cutoff={float(cutoff)}"
        )
    return best[2]


def _classify_valid_space_zone_totals(
    zone_totals: list[int], *, box: float, cutoff: float
) -> tuple[list[int], list[dict[str, object]], list[str]]:
    valid_zone_totals: list[int] = []
    skipped: list[dict[str, object]] = []
    layout_specs: list[str] = []
    for zones_total in zone_totals:
        try:
            nx, ny, nz = _best_space_layout(int(zones_total), float(box), float(cutoff))
        except ValueError as exc:
            skipped.append(
                {
                    "zones_total": int(zones_total),
                    "reason": str(exc),
                }
            )
            continue
        valid_zone_totals.append(int(zones_total))
        layout_specs.append(f"{int(zones_total)}:{int(nx)}x{int(ny)}x{int(nz)}")
    return valid_zone_totals, skipped, layout_specs


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _lookup_case(
    summary: dict[str, object], *, kind: str, device: str, zones_total: int
) -> dict[str, object] | None:
    by_case = dict(summary.get("by_case", {}) or {})
    exact_key = f"{kind}_{'gpu' if str(device) == 'cuda' else 'cpu'}_z{int(zones_total)}"
    case = dict(by_case.get(exact_key, {}) or {})
    if case:
        return case
    suffix = f"_z{int(zones_total)}"
    prefix = f"{kind}_"
    for key, value in by_case.items():
        key_str = str(key)
        if key_str.startswith(prefix) and key_str.endswith(suffix):
            case = dict(value or {})
            if case:
                return case
    return None


def _run_subprocess(*, cmd: list[str], timeout: int) -> tuple[int, str, str, bool]:
    timed_out = False
    proc_rc = 1
    proc_out = ""
    proc_err = ""
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout)),
        )
        proc_rc = int(proc.returncode)
        proc_out = str(proc.stdout or "")
        proc_err = str(proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        proc_rc = 124
        proc_out = str(getattr(exc, "stdout", "") or "")
        proc_err = str(getattr(exc, "stderr", "") or "")
    return proc_rc, proc_out, proc_err, timed_out


def _write_crack_task(
    *,
    task_path: Path,
    target_atoms: int,
    box: float,
    lattice_a: float,
    dt: float,
    steps: int,
    cutoff: float,
    eam_file: str,
    seed: int,
    velocity_std: float,
) -> dict[str, object]:
    positions, types, masses, velocities, crack_lo, crack_hi, available_after_crack = (
        build_al_crack_state(
            target_atoms=int(target_atoms),
            box=float(box),
            lattice_a=float(lattice_a),
            seed=int(seed),
            velocity_std=float(velocity_std),
        )
    )
    write_al_crack_task_yaml(
        out_path=task_path,
        box=float(box),
        dt=float(dt),
        steps=int(steps),
        cutoff=float(cutoff),
        positions=positions,
        types=types,
        masses=masses,
        velocities=velocities,
        eam_file=str(eam_file),
    )
    return {
        "path": str(task_path),
        "available_after_crack": int(available_after_crack),
        "crack_geometry": {
            "lo": [float(v) for v in crack_lo],
            "hi": [float(v) for v in crack_hi],
        },
    }


def _run_exact_crack_compare(
    *,
    task_path: Path,
    out_dir: Path,
    device: str,
    requested_zones: int,
    cell_size: float,
    timeout_sec: int,
    requested_space_timeout_sec: int,
    compare_space_timeout_sec: int,
    compare_time_timeout_sec: int,
    telemetry_every: int,
    telemetry_heartbeat_sec: float,
    telemetry_stdout: bool,
    require_effective_cuda: bool,
) -> dict[str, object]:
    out_csv = out_dir / "al_crack_100k_compare_gpu.csv"
    out_md = out_dir / "al_crack_100k_compare_gpu.md"
    out_json = out_dir / "al_crack_100k_compare_gpu.summary.json"
    telemetry_dir = out_dir / "al_crack_100k_compare_gpu_telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/bench_al_crack_compare.py",
        "--task",
        str(task_path),
        "--out",
        str(out_csv),
        "--md",
        str(out_md),
        "--json",
        str(out_json),
        "--device",
        str(device),
        "--requested-zones",
        str(int(requested_zones)),
        "--cell-size",
        str(float(cell_size)),
        "--timeout-sec",
        str(int(timeout_sec)),
        "--requested-space-timeout-sec",
        str(int(requested_space_timeout_sec)),
        "--compare-space-timeout-sec",
        str(int(compare_space_timeout_sec)),
        "--compare-time-timeout-sec",
        str(int(compare_time_timeout_sec)),
        "--telemetry-dir",
        str(telemetry_dir),
        "--telemetry-every",
        str(int(telemetry_every)),
        "--telemetry-heartbeat-sec",
        str(float(telemetry_heartbeat_sec)),
    ]
    if bool(telemetry_stdout):
        cmd.append("--telemetry-stdout")
    if bool(require_effective_cuda):
        cmd.append("--require-effective-cuda")
    proc_rc, proc_out, proc_err, timed_out = _run_subprocess(cmd=cmd, timeout=int(timeout_sec) + 30)
    summary = _load_json(out_json)
    benchmark_ok_all = bool(summary.get("ok_all", False))
    requested_td_preflight = dict(summary.get("requested_td_preflight", {}) or {})
    requested_time_required = bool(requested_td_preflight.get("preflight_ok", False))
    common_zones_total = int(summary.get("strict_valid_common_zones_total", 0) or 0)
    requested_space_case = _lookup_case(
        summary, kind="space", device=str(device), zones_total=int(requested_zones)
    )
    common_space_case = _lookup_case(
        summary, kind="space", device=str(device), zones_total=int(common_zones_total)
    )
    common_time_case = _lookup_case(
        summary, kind="time", device=str(device), zones_total=int(common_zones_total)
    )
    requested_time_case = (
        _lookup_case(summary, kind="time", device=str(device), zones_total=int(requested_zones))
        if requested_time_required
        else None
    )
    executed_cases: list[dict[str, object]] = []
    for case in (
        requested_space_case,
        common_space_case,
        common_time_case,
        requested_time_case,
    ):
        if case:
            executed_cases.append(case)
    executed_cuda_ok = True
    if bool(require_effective_cuda):
        executed_cuda_ok = bool(
            executed_cases
            and all(str(case.get("effective_device", "cpu")) == "cuda" for case in executed_cases)
        )
    evidence_ok = bool(
        int(proc_rc) == 0
        and not timed_out
        and int(summary.get("requested_zones_total", 0) or 0) == int(requested_zones)
        and int(common_zones_total) > 0
        and isinstance(summary.get("requested_td_wavefront", {}), dict)
        and isinstance(summary.get("strict_valid_common_td_wavefront", {}), dict)
        and requested_space_case is not None
        and common_space_case is not None
        and bool(common_space_case.get("ok", False))
        and common_time_case is not None
        and bool(common_time_case.get("ok", False))
        and (not requested_time_required or bool((requested_time_case or {}).get("ok", False)))
        and executed_cuda_ok
    )
    summary["external_returncode"] = int(proc_rc)
    summary["external_stdout"] = proc_out
    summary["external_stderr"] = proc_err
    summary["external_timed_out"] = bool(timed_out)
    summary["benchmark_ok_all"] = bool(benchmark_ok_all)
    summary["requested_space_case_ok"] = bool((requested_space_case or {}).get("ok", False))
    summary["strict_valid_common_space_case_ok"] = bool((common_space_case or {}).get("ok", False))
    summary["strict_valid_common_time_case_ok"] = bool((common_time_case or {}).get("ok", False))
    summary["requested_td_case_required"] = bool(requested_time_required)
    summary["requested_td_case_ok"] = bool(
        (not requested_time_required) or bool((requested_time_case or {}).get("ok", False))
    )
    summary["ok_all"] = bool(evidence_ok)
    return summary


def _run_crack_sweep(
    *,
    task_path: Path,
    out_dir: Path,
    device: str,
    zone_totals: list[int],
    cell_size: float,
    per_zone_timeout_sec: int,
    requested_space_timeout_sec: int,
    compare_space_timeout_sec: int,
    compare_time_timeout_sec: int,
    telemetry_every: int,
    telemetry_heartbeat_sec: float,
    require_effective_cuda: bool,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    by_zone: dict[str, dict[str, object]] = {}
    for zones_total in zone_totals:
        case_stem = f"z{int(zones_total)}"
        out_csv = out_dir / f"{case_stem}.csv"
        out_md = out_dir / f"{case_stem}.md"
        out_json = out_dir / f"{case_stem}.summary.json"
        telemetry_dir = out_dir / f"{case_stem}_telemetry"
        cmd = [
            sys.executable,
            "scripts/bench_al_crack_compare.py",
            "--task",
            str(task_path),
            "--out",
            str(out_csv),
            "--md",
            str(out_md),
            "--json",
            str(out_json),
            "--device",
            str(device),
            "--requested-zones",
            str(int(zones_total)),
            "--cell-size",
            str(float(cell_size)),
            "--timeout-sec",
            str(int(per_zone_timeout_sec)),
            "--requested-space-timeout-sec",
            str(int(requested_space_timeout_sec)),
            "--compare-space-timeout-sec",
            str(int(compare_space_timeout_sec)),
            "--compare-time-timeout-sec",
            str(int(compare_time_timeout_sec)),
            "--telemetry-dir",
            str(telemetry_dir),
            "--telemetry-every",
            str(int(telemetry_every)),
            "--telemetry-heartbeat-sec",
            str(float(telemetry_heartbeat_sec)),
            "--strict",
        ]
        if bool(require_effective_cuda):
            cmd.append("--require-effective-cuda")
        proc_rc, proc_out, proc_err, timed_out = _run_subprocess(
            cmd=cmd, timeout=int(per_zone_timeout_sec) + 30
        )
        summary = _load_json(out_json)
        by_case = dict(summary.get("by_case", {}) or {})
        space_case = dict(
            by_case.get(
                f"space_{'gpu' if str(device) == 'cuda' else 'cpu'}_z{int(zones_total)}", {}
            )
            or {}
        )
        time_case = dict(
            by_case.get(f"time_{'gpu' if str(device) == 'cuda' else 'cpu'}_z{int(zones_total)}", {})
            or {}
        )
        if not space_case:
            for key, value in by_case.items():
                if str(key).startswith("space_") and str(key).endswith(f"_z{int(zones_total)}"):
                    space_case = dict(value or {})
                    break
        if not time_case:
            for key, value in by_case.items():
                if str(key).startswith("time_") and str(key).endswith(f"_z{int(zones_total)}"):
                    time_case = dict(value or {})
                    break
        row = {
            "zones_total": int(zones_total),
            "ok_all": bool(summary.get("ok_all", False) and int(proc_rc) == 0 and not timed_out),
            "effective_device": str(summary.get("effective_device", "")),
            "time_gpu_elapsed_sec": time_case.get("elapsed_sec"),
            "space_gpu_elapsed_sec": space_case.get("elapsed_sec"),
            "time_gpu_steps_per_sec": time_case.get("steps_per_sec"),
            "space_gpu_steps_per_sec": space_case.get("steps_per_sec"),
            "td_speedup_vs_space": dict(summary.get("comparisons", {}) or {}).get(
                "exact_request_td_speedup_vs_space"
            ),
            "used_common_fallback": int(
                not bool(
                    dict(summary.get("requested_td_preflight", {}) or {}).get("preflight_ok", False)
                )
            ),
            "wavefront_first_wave_size": int(
                dict(summary.get("requested_td_wavefront", {}) or {}).get("first_wave_size", 0) or 0
            ),
            "wavefront_deferred_zones_total": int(
                dict(summary.get("requested_td_wavefront", {}) or {}).get("deferred_zones_total", 0)
                or 0
            ),
            "wavefront_fallback_to_sequential_reasons": ",".join(
                str(item)
                for item in dict(summary.get("requested_td_wavefront", {}) or {}).get(
                    "fallback_to_sequential_reasons", []
                )
                or []
            ),
            "summary_json": str(out_json),
            "summary_md": str(out_md),
            "telemetry_dir": str(telemetry_dir),
            "external_returncode": int(proc_rc),
            "external_timed_out": bool(timed_out),
            "external_stdout": proc_out,
            "external_stderr": proc_err,
        }
        rows.append(row)
        by_zone[str(int(zones_total))] = {
            "summary": summary,
            "row": row,
        }

    ok_rows = [row for row in rows if bool(row.get("ok_all", False))]
    td_runtime = [
        {
            "zones_total": int(row["zones_total"]),
            "time_gpu_elapsed_sec": row.get("time_gpu_elapsed_sec"),
        }
        for row in rows
    ]
    paired = [
        {
            "zones_total": int(row["zones_total"]),
            "space_gpu_elapsed_sec": row.get("space_gpu_elapsed_sec"),
            "time_gpu_elapsed_sec": row.get("time_gpu_elapsed_sec"),
            "td_speedup_vs_space": row.get("td_speedup_vs_space"),
        }
        for row in rows
    ]
    return {
        "n": len(rows),
        "total": len(rows),
        "ok": int(sum(1 for row in rows if bool(row.get("ok_all", False)))),
        "fail": int(sum(1 for row in rows if not bool(row.get("ok_all", False)))),
        "ok_all": all(bool(row.get("ok_all", False)) for row in rows),
        "rows": rows,
        "by_zone": by_zone,
        "td_absolute_runtime_vs_z": td_runtime,
        "equal_z_td_vs_space": paired,
        "best_td_speedup_row": (
            max(ok_rows, key=lambda row: float(row.get("td_speedup_vs_space", 0.0) or 0.0))
            if ok_rows
            else None
        ),
        "fastest_td_row": (
            min(
                ok_rows,
                key=lambda row: float(
                    row.get("time_gpu_elapsed_sec", float("inf")) or float("inf")
                ),
            )
            if ok_rows
            else None
        ),
        "artifacts": {"dir": str(out_dir)},
    }


def _run_control_sweep(
    *,
    out_dir: Path,
    n_atoms: int,
    steps: int,
    repeats: int,
    warmup: int,
    seed: int,
    cutoff: float,
    dt: float,
    cell_size: float,
    zone_cells_w: int,
    zone_cells_s: int,
    zone_totals: list[int],
    eam_file: str,
    require_effective_cuda: bool,
) -> dict[str, object]:
    _, _, _, _, control_box = _build_alloy_state(
        n_atoms=int(n_atoms),
        lattice_a=4.05,
        jitter=0.02,
        seed=int(seed),
        velocity_std=0.01,
    )
    valid_zone_totals, skipped_zone_totals, layouts = _classify_valid_space_zone_totals(
        zone_totals,
        box=float(control_box),
        cutoff=float(cutoff),
    )
    if not valid_zone_totals:
        raise ValueError("no valid control-sweep space layouts for the requested zone totals")
    out_csv = out_dir / "eam_decomp_zone_sweep_gpu.csv"
    out_md = out_dir / "eam_decomp_zone_sweep_gpu.md"
    out_json = out_dir / "eam_decomp_zone_sweep_gpu.summary.json"
    cmd = [
        sys.executable,
        "scripts/bench_eam_zone_sweep_gpu.py",
        "--out",
        str(out_csv),
        "--md",
        str(out_md),
        "--json",
        str(out_json),
        "--n-atoms",
        str(int(n_atoms)),
        "--steps",
        str(int(steps)),
        "--repeats",
        str(int(repeats)),
        "--warmup",
        str(int(warmup)),
        "--seed",
        str(int(seed)),
        "--cutoff",
        str(float(cutoff)),
        "--dt",
        str(float(dt)),
        "--cell-size",
        str(float(cell_size)),
        "--zone-cells-w",
        str(int(zone_cells_w)),
        "--zone-cells-s",
        str(int(zone_cells_s)),
        "--layouts",
        ",".join(layouts),
        "--eam-file",
        str(eam_file),
        "--strict",
    ]
    if bool(require_effective_cuda):
        cmd.append("--require-effective-cuda")
    proc_rc, proc_out, proc_err, timed_out = _run_subprocess(cmd=cmd, timeout=900)
    summary = _load_json(out_json)
    summary["requested_zone_totals"] = [int(item) for item in zone_totals]
    summary["valid_zone_totals"] = [int(item) for item in valid_zone_totals]
    summary["skipped_invalid_space_zone_totals"] = skipped_zone_totals
    summary["external_returncode"] = int(proc_rc)
    summary["external_stdout"] = proc_out
    summary["external_stderr"] = proc_err
    summary["external_timed_out"] = bool(timed_out)
    summary["ok_all"] = bool(summary.get("ok_all", False) and int(proc_rc) == 0 and not timed_out)
    summary["td_absolute_runtime_vs_z"] = [
        {
            "zones_total": int(row.get("zones_total", 0) or 0),
            "time_gpu_median_sec": row.get("time_gpu_median_sec"),
        }
        for row in list(summary.get("rows", []))
    ]
    summary["equal_z_td_vs_space"] = [
        {
            "zones_total": int(row.get("zones_total", 0) or 0),
            "space_gpu_median_sec": row.get("space_gpu_median_sec"),
            "time_gpu_median_sec": row.get("time_gpu_median_sec"),
            "td_speedup_vs_space": row.get("td_speedup_vs_space"),
        }
        for row in list(summary.get("rows", []))
    ]
    return summary


def _run_control_breakdown(
    *,
    out_dir: Path,
    n_atoms: int,
    steps: int,
    repeats: int,
    warmup: int,
    seed: int,
    cutoff: float,
    dt: float,
    cell_size: float,
    zone_cells_w: int,
    zone_cells_s: int,
    zones_total: int,
    eam_file: str,
    require_effective_cuda: bool,
) -> dict[str, object]:
    _, _, _, _, control_box = _build_alloy_state(
        n_atoms=int(n_atoms),
        lattice_a=4.05,
        jitter=0.02,
        seed=int(seed),
        velocity_std=0.01,
    )
    nx, ny, nz = _best_space_layout(int(zones_total), float(control_box), float(cutoff))
    out_csv = out_dir / "eam_td_breakdown_gpu.csv"
    out_md = out_dir / "eam_td_breakdown_gpu.md"
    out_json = out_dir / "eam_td_breakdown_gpu.summary.json"
    cmd = [
        sys.executable,
        "scripts/bench_eam_td_breakdown_gpu.py",
        "--out",
        str(out_csv),
        "--md",
        str(out_md),
        "--json",
        str(out_json),
        "--n-atoms",
        str(int(n_atoms)),
        "--steps",
        str(int(steps)),
        "--repeats",
        str(int(repeats)),
        "--warmup",
        str(int(warmup)),
        "--seed",
        str(int(seed)),
        "--cutoff",
        str(float(cutoff)),
        "--dt",
        str(float(dt)),
        "--cell-size",
        str(float(cell_size)),
        "--zones-total",
        str(int(zones_total)),
        "--zone-cells-w",
        str(int(zone_cells_w)),
        "--zone-cells-s",
        str(int(zone_cells_s)),
        "--zones-nx",
        str(int(nx)),
        "--zones-ny",
        str(int(ny)),
        "--zones-nz",
        str(int(nz)),
        "--eam-file",
        str(eam_file),
        "--strict",
    ]
    if bool(require_effective_cuda):
        cmd.append("--require-effective-cuda")
    proc_rc, proc_out, proc_err, timed_out = _run_subprocess(cmd=cmd, timeout=900)
    summary = _load_json(out_json)
    time_gpu = dict(dict(summary.get("by_case", {}) or {}).get("time_gpu", {}) or {})
    time_breakdown = dict(time_gpu.get("breakdown", {}) or {})
    summary["selected_zones_total"] = int(zones_total)
    summary["selected_space_layout"] = {"nx": int(nx), "ny": int(ny), "nz": int(nz)}
    summary["external_returncode"] = int(proc_rc)
    summary["external_stdout"] = proc_out
    summary["external_stderr"] = proc_err
    summary["external_timed_out"] = bool(timed_out)
    summary["ok_all"] = bool(summary.get("ok_all", False) and int(proc_rc) == 0 and not timed_out)
    summary["time_gpu_force_kernel_share"] = float(
        (time_breakdown.get("forces_full_share", 0.0) or 0.0)
        + (time_breakdown.get("target_local_force_share", 0.0) or 0.0)
    )
    summary["time_gpu_orchestration_share"] = float(
        (time_breakdown.get("device_sync_share", 0.0) or 0.0)
        + (time_breakdown.get("build_cell_list_share", 0.0) or 0.0)
        + (time_breakdown.get("zone_buffer_skin_share", 0.0) or 0.0)
        + (time_breakdown.get("candidate_enum_share", 0.0) or 0.0)
        + (time_breakdown.get("zone_assign_share", 0.0) or 0.0)
        + (time_breakdown.get("other_share", 0.0) or 0.0)
    )
    summary["time_gpu_wave_batch_launches_saved_per_step"] = float(
        dict(time_breakdown.get("wave_batch_diagnostics", {}) or {}).get("launches_saved_per_step", 0.0)
        or 0.0
    )
    summary["time_gpu_wave_batch_neighbor_reuse_ratio"] = float(
        dict(time_breakdown.get("wave_batch_diagnostics", {}) or {}).get(
            "neighbor_reuse_ratio_weighted", 0.0
        )
        or 0.0
    )
    summary["time_gpu_wave_batch_candidate_union_to_target_ratio"] = float(
        dict(time_breakdown.get("wave_batch_diagnostics", {}) or {}).get(
            "candidate_union_to_target_ratio_avg", 0.0
        )
        or 0.0
    )
    return summary


def _build_report(
    *,
    exact_compare: dict[str, object],
    crack_sweep: dict[str, object],
    control_sweep: dict[str, object],
    control_breakdown: dict[str, object],
) -> str:
    def _fmt_speedup(value: object) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.3f}x"

    lines = [
        "# Slab Wavefront Evidence Pack",
        "",
        "## Exact Crack Benchmark",
        "",
        f"- evidence_ok: `{bool(exact_compare.get('ok_all', False))}`",
        f"- child_benchmark_ok_all: `{bool(exact_compare.get('benchmark_ok_all', False))}`",
        f"- requested_zones_total: `{int(exact_compare.get('requested_zones_total', 0) or 0)}`",
        f"- strict_valid_common_zones_total: `{int(exact_compare.get('strict_valid_common_zones_total', 0) or 0)}`",
        f"- requested_space_case_ok: `{bool(exact_compare.get('requested_space_case_ok', False))}`",
        f"- strict_valid_common_space_case_ok: `{bool(exact_compare.get('strict_valid_common_space_case_ok', False))}`",
        f"- strict_valid_common_time_case_ok: `{bool(exact_compare.get('strict_valid_common_time_case_ok', False))}`",
        f"- exact_request_td_speedup_vs_space: `{_fmt_speedup(dict(exact_compare.get('comparisons', {}) or {}).get('exact_request_td_speedup_vs_space'))}`",
        f"- strict_valid_td_speedup_vs_space: `{_fmt_speedup(dict(exact_compare.get('comparisons', {}) or {}).get('strict_valid_td_speedup_vs_space'))}`",
        "",
        "## Crack Sweep `z` Trend",
        "",
        "| zones_total | space_gpu_sec | time_gpu_sec | td_speedup_vs_space | wavefront_first_wave | wavefront_deferred | wave_launches_saved/step |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in list(crack_sweep.get("rows", [])):
        lines.append(
            "| "
            + str(int(row.get("zones_total", 0) or 0))
            + " | "
            + (
                f"{float(row.get('space_gpu_elapsed_sec', 0.0) or 0.0):.6f}"
                if row.get("space_gpu_elapsed_sec") is not None
                else "n/a"
            )
            + " | "
            + (
                f"{float(row.get('time_gpu_elapsed_sec', 0.0) or 0.0):.6f}"
                if row.get("time_gpu_elapsed_sec") is not None
                else "n/a"
            )
            + " | "
            + (
                f"{float(row.get('td_speedup_vs_space', 0.0) or 0.0):.6f}"
                if row.get("td_speedup_vs_space") is not None
                else "n/a"
            )
            + " | "
            + str(int(row.get("wavefront_first_wave_size", 0) or 0))
            + " | "
            + str(int(row.get("wavefront_deferred_zones_total", 0) or 0))
            + " | "
            + f"{float(row.get('time_gpu_wave_batch_launches_saved_per_step', 0.0) or 0.0):.3f}"
            + " |"
        )

    lines.extend(
        [
            "",
            "## EAM Control Sweep",
            "",
        ]
    )
    skipped_control = list(control_sweep.get("skipped_invalid_space_zone_totals", []) or [])
    if skipped_control:
        skipped_txt = ", ".join(
            f"z={int(dict(item or {}).get('zones_total', 0) or 0)}"
            for item in skipped_control
        )
        lines.extend(
            [
                f"- skipped_invalid_space_zone_totals: `{skipped_txt}`",
                "",
            ]
        )
    lines.extend(
        [
            "| zones_total | space_gpu_sec | time_gpu_sec | td_speedup_vs_space | wavefront_first_wave | wavefront_deferred | wave_launches_saved/step |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in list(control_sweep.get("rows", [])):
        lines.append(
            "| "
            + str(int(row.get("zones_total", 0) or 0))
            + " | "
            + f"{float(row.get('space_gpu_median_sec', 0.0) or 0.0):.6f}"
            + " | "
            + f"{float(row.get('time_gpu_median_sec', 0.0) or 0.0):.6f}"
            + " | "
            + (
                f"{float(row.get('td_speedup_vs_space', 0.0) or 0.0):.6f}"
                if row.get("td_speedup_vs_space") is not None
                else "n/a"
            )
            + " | "
            + str(int(row.get("wavefront_first_wave_size", 0) or 0))
            + " | "
            + str(int(row.get("wavefront_deferred_zones_total", 0) or 0))
            + " | "
            + f"{float(row.get('time_gpu_wave_batch_launches_saved_per_step', 0.0) or 0.0):.3f}"
            + " |"
        )

    time_gpu = dict(dict(control_breakdown.get("by_case", {}) or {}).get("time_gpu", {}) or {})
    time_breakdown = dict(time_gpu.get("breakdown", {}) or {})
    lines.extend(
        [
            "",
            "## Control Breakdown",
            "",
            f"- selected_zones_total: `{int(control_breakdown.get('selected_zones_total', 0) or 0)}`",
            f"- selected_space_layout: `{int(dict(control_breakdown.get('selected_space_layout', {}) or {}).get('nx', 0))}x{int(dict(control_breakdown.get('selected_space_layout', {}) or {}).get('ny', 0))}x{int(dict(control_breakdown.get('selected_space_layout', {}) or {}).get('nz', 0))}`",
            f"- td_speedup_gpu: `{float(dict(control_breakdown.get('comparisons', {}) or {}).get('td_speedup_gpu', 0.0) or 0.0):.3f}x`",
            f"- time_gpu_force_kernel_share: `{100.0 * float(control_breakdown.get('time_gpu_force_kernel_share', 0.0) or 0.0):.2f}%`",
            f"- time_gpu_orchestration_share: `{100.0 * float(control_breakdown.get('time_gpu_orchestration_share', 0.0) or 0.0):.2f}%`",
            f"- time_gpu_device_sync_share: `{100.0 * float(time_breakdown.get('device_sync_share', 0.0) or 0.0):.2f}%`",
            f"- time_gpu_candidate_enum_share: `{100.0 * float(time_breakdown.get('candidate_enum_share', 0.0) or 0.0):.2f}%`",
            f"- time_gpu_zone_assign_share: `{100.0 * float(time_breakdown.get('zone_assign_share', 0.0) or 0.0):.2f}%`",
            f"- time_gpu_wave_batch_launches_saved_per_step: `{float(control_breakdown.get('time_gpu_wave_batch_launches_saved_per_step', 0.0) or 0.0):.3f}`",
            f"- time_gpu_wave_batch_neighbor_reuse: `{100.0 * float(control_breakdown.get('time_gpu_wave_batch_neighbor_reuse_ratio', 0.0) or 0.0):.2f}%`",
            f"- time_gpu_wave_batch_union_to_target_ratio: `{float(control_breakdown.get('time_gpu_wave_batch_candidate_union_to_target_ratio', 0.0) or 0.0):.3f}`",
            "",
            "## Interpretation",
            "",
            "- Crack sweep rows distinguish TD-vs-space at equal `z` and TD absolute runtime vs `z`.",
            "- EAM control sweep provides the same separation on a non-crack workload.",
            "- Control breakdown separates force-kernel share from orchestration/neighbor/sync overhead for one representative `z` and records realized runtime wave-batch savings.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Representative evidence pack for single-GPU 1D slab wavefront viability"
    )
    ap.add_argument("--out", default="results/slab_wavefront_evidence_gpu.csv")
    ap.add_argument("--md", default="results/slab_wavefront_evidence_gpu.md")
    ap.add_argument("--json", default="results/slab_wavefront_evidence_gpu.summary.json")
    ap.add_argument("--target-atoms", type=int, default=100000)
    ap.add_argument("--box", type=float, default=122.0)
    ap.add_argument("--lattice-a", type=float, default=4.05)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cutoff", type=float, default=6.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--exact-requested-zones", type=int, default=1000)
    ap.add_argument("--crack-zones", default="1,2,3,4,5,6,7,8,9,10,11,12")
    ap.add_argument("--control-zone-totals", default="1,2,3,4,5,6,7,8,9,10,11,12")
    ap.add_argument("--control-breakdown-zones-total", type=int, default=0)
    ap.add_argument("--cell-size", type=float, default=0.5)
    ap.add_argument("--control-n-atoms", type=int, default=10000)
    ap.add_argument("--control-steps", type=int, default=256)
    ap.add_argument("--control-repeats", type=int, default=1)
    ap.add_argument("--control-warmup", type=int, default=1)
    ap.add_argument("--zone-cells-w", type=int, default=1)
    ap.add_argument("--zone-cells-s", type=int, default=2)
    ap.add_argument(
        "--eam-file",
        default="examples/potentials/eam_alloy/Al_zhou.eam.alloy",
        help="Single-element Al EAM file used in the generated crack task",
    )
    ap.add_argument(
        "--control-eam-file",
        default="examples/potentials/eam_alloy/AlCu.eam.alloy",
        help="EAM/alloy file used by the control sweep/breakdown scripts",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--velocity-std", type=float, default=0.020)
    ap.add_argument("--exact-timeout-sec", type=int, default=600)
    ap.add_argument("--crack-sweep-timeout-sec", type=int, default=180)
    ap.add_argument("--requested-space-timeout-sec", type=int, default=180)
    ap.add_argument("--compare-space-timeout-sec", type=int, default=180)
    ap.add_argument("--compare-time-timeout-sec", type=int, default=180)
    ap.add_argument("--telemetry-every", type=int, default=1)
    ap.add_argument("--telemetry-heartbeat-sec", type=float, default=5.0)
    ap.add_argument("--telemetry-stdout", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--require-effective-cuda", action="store_true")
    args = ap.parse_args()

    crack_zones = _parse_int_list(str(args.crack_zones))
    control_zone_totals = _parse_int_list(str(args.control_zone_totals))
    breakdown_zones_total = (
        int(args.control_breakdown_zones_total)
        if int(args.control_breakdown_zones_total) > 0
        else max(control_zone_totals)
    )

    out_csv = Path(args.out)
    out_md = Path(args.md)
    out_json = Path(args.json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    base_dir = out_json.parent / out_json.stem
    base_dir.mkdir(parents=True, exist_ok=True)

    task_info = _write_crack_task(
        task_path=base_dir / "slab_wavefront_evidence_gpu.task.yaml",
        target_atoms=int(args.target_atoms),
        box=float(args.box),
        lattice_a=float(args.lattice_a),
        dt=float(args.dt),
        steps=int(args.steps),
        cutoff=float(args.cutoff),
        eam_file=str(args.eam_file),
        seed=int(args.seed),
        velocity_std=float(args.velocity_std),
    )
    task_path = Path(str(task_info["path"]))

    exact_compare = _run_exact_crack_compare(
        task_path=task_path,
        out_dir=base_dir / "exact_compare",
        device=str(args.device),
        requested_zones=int(args.exact_requested_zones),
        cell_size=float(args.cell_size),
        timeout_sec=int(args.exact_timeout_sec),
        requested_space_timeout_sec=int(args.requested_space_timeout_sec),
        compare_space_timeout_sec=int(args.compare_space_timeout_sec),
        compare_time_timeout_sec=int(args.compare_time_timeout_sec),
        telemetry_every=int(args.telemetry_every),
        telemetry_heartbeat_sec=float(args.telemetry_heartbeat_sec),
        telemetry_stdout=bool(args.telemetry_stdout),
        require_effective_cuda=bool(args.require_effective_cuda),
    )
    crack_sweep = _run_crack_sweep(
        task_path=task_path,
        out_dir=base_dir / "crack_sweep",
        device=str(args.device),
        zone_totals=crack_zones,
        cell_size=float(args.cell_size),
        per_zone_timeout_sec=int(args.crack_sweep_timeout_sec),
        requested_space_timeout_sec=int(args.requested_space_timeout_sec),
        compare_space_timeout_sec=int(args.compare_space_timeout_sec),
        compare_time_timeout_sec=int(args.compare_time_timeout_sec),
        telemetry_every=int(args.telemetry_every),
        telemetry_heartbeat_sec=float(args.telemetry_heartbeat_sec),
        require_effective_cuda=bool(args.require_effective_cuda),
    )
    control_sweep = _run_control_sweep(
        out_dir=base_dir / "control_sweep",
        n_atoms=int(args.control_n_atoms),
        steps=int(args.control_steps),
        repeats=int(args.control_repeats),
        warmup=int(args.control_warmup),
        seed=int(args.seed),
        cutoff=float(args.cutoff),
        dt=float(args.dt),
        cell_size=max(2.0, float(args.cutoff) / 3.0),
        zone_cells_w=int(args.zone_cells_w),
        zone_cells_s=int(args.zone_cells_s),
        zone_totals=control_zone_totals,
        eam_file=str(args.control_eam_file),
        require_effective_cuda=bool(args.require_effective_cuda),
    )
    control_breakdown = _run_control_breakdown(
        out_dir=base_dir / "control_breakdown",
        n_atoms=int(args.control_n_atoms),
        steps=int(args.control_steps),
        repeats=int(args.control_repeats),
        warmup=int(args.control_warmup),
        seed=int(args.seed),
        cutoff=float(args.cutoff),
        dt=float(args.dt),
        cell_size=max(2.0, float(args.cutoff) / 3.0),
        zone_cells_w=int(args.zone_cells_w),
        zone_cells_s=int(args.zone_cells_s),
        zones_total=int(breakdown_zones_total),
        eam_file=str(args.control_eam_file),
        require_effective_cuda=bool(args.require_effective_cuda),
    )

    report_markdown = _build_report(
        exact_compare=exact_compare,
        crack_sweep=crack_sweep,
        control_sweep=control_sweep,
        control_breakdown=control_breakdown,
    )

    sections = {
        "exact_compare": exact_compare,
        "crack_sweep": crack_sweep,
        "control_sweep": control_sweep,
        "control_breakdown": control_breakdown,
    }
    ok_all = all(bool(dict(section or {}).get("ok_all", False)) for section in sections.values())

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "section",
                "zones_total",
                "space_gpu_sec",
                "time_gpu_sec",
                "td_speedup_vs_space",
                "wavefront_first_wave_size",
                "wavefront_deferred_zones_total",
                "wave_launches_saved_per_step",
                "artifact_json",
            ]
        )
        for row in list(crack_sweep.get("rows", [])):
            w.writerow(
                [
                    "crack_sweep",
                    int(row.get("zones_total", 0) or 0),
                    row.get("space_gpu_elapsed_sec", ""),
                    row.get("time_gpu_elapsed_sec", ""),
                    row.get("td_speedup_vs_space", ""),
                    int(row.get("wavefront_first_wave_size", 0) or 0),
                    int(row.get("wavefront_deferred_zones_total", 0) or 0),
                    float(row.get("time_gpu_wave_batch_launches_saved_per_step", 0.0) or 0.0),
                    row.get("summary_json", ""),
                ]
            )
        for row in list(control_sweep.get("rows", [])):
            w.writerow(
                [
                    "control_sweep",
                    int(row.get("zones_total", 0) or 0),
                    row.get("space_gpu_median_sec", ""),
                    row.get("time_gpu_median_sec", ""),
                    row.get("td_speedup_vs_space", ""),
                    int(row.get("wavefront_first_wave_size", 0) or 0),
                    int(row.get("wavefront_deferred_zones_total", 0) or 0),
                    float(row.get("time_gpu_wave_batch_launches_saved_per_step", 0.0) or 0.0),
                    str(Path(dict(control_sweep.get("artifacts", {}) or {}).get("json", out_json))),
                ]
            )

    out_md.write_text(report_markdown, encoding="utf-8")
    summary = {
        "n": len(sections),
        "total": len(sections),
        "ok": int(
            sum(
                1 for section in sections.values() if bool(dict(section or {}).get("ok_all", False))
            )
        ),
        "fail": int(
            sum(
                1
                for section in sections.values()
                if not bool(dict(section or {}).get("ok_all", False))
            )
        ),
        "ok_all": bool(ok_all),
        "task": task_info,
        "exact_compare": exact_compare,
        "crack_sweep": crack_sweep,
        "control_sweep": control_sweep,
        "control_breakdown": control_breakdown,
        "evidence_views": {
            "crack_equal_z_td_vs_space": list(crack_sweep.get("equal_z_td_vs_space", [])),
            "crack_td_absolute_runtime_vs_z": list(crack_sweep.get("td_absolute_runtime_vs_z", [])),
            "control_equal_z_td_vs_space": list(control_sweep.get("equal_z_td_vs_space", [])),
            "control_td_absolute_runtime_vs_z": list(
                control_sweep.get("td_absolute_runtime_vs_z", [])
            ),
            "control_orchestration_vs_force": {
                "selected_zones_total": int(control_breakdown.get("selected_zones_total", 0) or 0),
                "time_gpu_force_kernel_share": float(
                    control_breakdown.get("time_gpu_force_kernel_share", 0.0) or 0.0
                ),
                "time_gpu_orchestration_share": float(
                    control_breakdown.get("time_gpu_orchestration_share", 0.0) or 0.0
                ),
            },
            "control_wave_batch_cost_model": {
                "selected_zones_total": int(control_breakdown.get("selected_zones_total", 0) or 0),
                "launches_saved_per_step": float(
                    control_breakdown.get("time_gpu_wave_batch_launches_saved_per_step", 0.0)
                    or 0.0
                ),
                "neighbor_reuse_ratio": float(
                    control_breakdown.get("time_gpu_wave_batch_neighbor_reuse_ratio", 0.0) or 0.0
                ),
                "candidate_union_to_target_ratio": float(
                    control_breakdown.get(
                        "time_gpu_wave_batch_candidate_union_to_target_ratio", 0.0
                    )
                    or 0.0
                ),
            },
        },
        "report_markdown": report_markdown,
        "effective_cuda_required": bool(args.require_effective_cuda),
        "artifacts": {
            "csv": str(out_csv),
            "md": str(out_md),
            "json": str(out_json),
            "base_dir": str(base_dir),
        },
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(report_markdown, end="")
    print(f"[slab-wavefront-evidence-gpu] wrote {out_csv}")
    print(f"[slab-wavefront-evidence-gpu] wrote {out_md}")
    print(f"[slab-wavefront-evidence-gpu] wrote {out_json}")
    if bool(args.strict) and not bool(summary["ok_all"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
