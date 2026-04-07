#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np

from bench_eam_decomp_perf import (
    _build_alloy_state,
    _fmt_speedup,
    _fmt_steps_per_sec,
    _fmt_time,
    _run_case as _run_perf_case,
)
from bench_eam_td_breakdown_gpu import _run_case as _run_breakdown_case
from tdmd.backend import resolve_backend
from tdmd.constants import GEOM_EPSILON
from tdmd.potentials import make_potential
from tdmd.wavefront_1d import describe_wavefront_1d_layout
from tdmd.zones import ZoneLayout1DCells


def _decode_name(value) -> str:
    if isinstance(value, bytes):
        return value.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    return str(value)


def _detect_resources() -> dict[str, object]:
    cpu_count = int(os.cpu_count() or 1)
    omp_threads = int(os.environ.get("OMP_NUM_THREADS", "0") or 0)
    mpi_world_size = int(
        os.environ.get("OMPI_COMM_WORLD_SIZE")
        or os.environ.get("PMI_SIZE")
        or os.environ.get("SLURM_NTASKS")
        or "1"
    )
    mpi_launchers = [name for name in ("mpirun", "mpiexec") if shutil.which(name)]

    backend = resolve_backend("cuda")
    effective_device = str(getattr(backend, "device", "cpu"))
    gpu_count = 0
    gpu_names: list[str] = []
    if effective_device == "cuda":
        try:
            cp = backend.xp
            gpu_count = int(cp.cuda.runtime.getDeviceCount())
            for idx in range(gpu_count):
                props = cp.cuda.runtime.getDeviceProperties(idx)
                gpu_names.append(_decode_name(props.get("name", f"gpu{idx}")))
        except Exception:
            gpu_count = 1
            gpu_names = ["cuda"]

    resource_parallel_budget = max(
        2,
        min(
            16,
            max(
                (gpu_count * 8) if gpu_count > 0 else 0,
                min(cpu_count, 12),
                max(2, mpi_world_size),
            ),
        ),
    )
    return {
        "cpu_count": cpu_count,
        "omp_threads_env": omp_threads,
        "mpi_world_size_env": mpi_world_size,
        "mpi_launchers": mpi_launchers,
        "requested_gpu_backend": "cuda",
        "effective_gpu_backend": effective_device,
        "gpu_count": int(gpu_count),
        "gpu_names": gpu_names,
        "resource_parallel_budget": int(resource_parallel_budget),
    }


def _factorizations(total: int) -> list[tuple[int, int, int]]:
    out: set[tuple[int, int, int]] = set()
    for nx in range(1, int(total) + 1):
        if int(total) % nx != 0:
            continue
        rem_xy = int(total) // nx
        for ny in range(1, rem_xy + 1):
            if rem_xy % ny != 0:
                continue
            nz = rem_xy // ny
            out.add(tuple(sorted((int(nx), int(ny), int(nz)), reverse=True)))
    return sorted(out)


def _best_space_layout(
    *, zones_total: int, box: float, cutoff: float
) -> tuple[int, int, int] | None:
    best: tuple[float, float, tuple[int, int, int]] | None = None
    for nx, ny, nz in _factorizations(int(zones_total)):
        widths = [float(box) / float(nx), float(box) / float(ny), float(box) / float(nz)]
        min_width = min(widths)
        max_width = max(widths)
        if min_width + GEOM_EPSILON < float(cutoff):
            continue
        balance = min_width / max_width if max_width > 0.0 else 0.0
        score = (float(balance), float(min_width))
        if best is None or score > (best[0], best[1]):
            best = (score[0], score[1], (int(nx), int(ny), int(nz)))
    return None if best is None else best[2]


def _is_valid_time_layout(
    *,
    box: float,
    cell_size: float,
    cutoff: float,
    zones_total: int,
    zone_cells_w: int,
    zone_cells_s: int,
) -> bool:
    try:
        ZoneLayout1DCells(
            box=float(box),
            cell_size=float(cell_size),
            zones_total=int(zones_total),
            zone_cells_w=int(zone_cells_w),
            zone_cells_s=int(zone_cells_s),
            min_zone_width=float(cutoff),
            strict_min_width=True,
        ).build()
    except Exception:
        return False
    return True


def _default_zone_totals(parallel_budget: int) -> list[int]:
    ladder = [2, 3, 4, 6, 8, 12, 16]
    return [value for value in ladder if int(value) <= int(parallel_budget)]


def _parse_zone_totals(zone_totals_arg: str) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for part in str(zone_totals_arg).split(","):
        txt = str(part).strip()
        if not txt:
            continue
        value = int(txt)
        if value <= 0:
            raise ValueError(f"zone total must be positive: {txt!r}")
        if value not in seen:
            out.append(value)
            seen.add(value)
    if not out:
        raise ValueError("at least one zone total must be provided")
    return out


def _enumerate_candidate_layouts(
    *,
    box: float,
    cutoff: float,
    cell_size: float,
    zone_cells_w: int,
    zone_cells_s: int,
    zone_totals: list[int],
) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for zones_total in zone_totals:
        if not _is_valid_time_layout(
            box=box,
            cell_size=cell_size,
            cutoff=cutoff,
            zones_total=zones_total,
            zone_cells_w=zone_cells_w,
            zone_cells_s=zone_cells_s,
        ):
            continue
        layout3 = _best_space_layout(zones_total=zones_total, box=box, cutoff=cutoff)
        if layout3 is None:
            continue
        nx, ny, nz = layout3
        out.append((int(zones_total), int(nx), int(ny), int(nz)))
    if not out:
        raise ValueError(
            "no strict-valid zone layouts were found for the current geometry/resources"
        )
    return out


def _layout_summary_rows(
    *,
    layouts: list[tuple[int, int, int, int]],
    r0,
    v0,
    masses,
    atom_types,
    box: float,
    potential,
    dt: float,
    cutoff: float,
    steps: int,
    repeats: int,
    warmup: int,
    cell_size: float,
    zone_cells_w: int,
    zone_cells_s: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for zones_total, zones_nx, zones_ny, zones_nz in layouts:
        wavefront = describe_wavefront_1d_layout(
            box=float(box),
            cutoff=float(cutoff),
            cell_size=float(cell_size),
            zones_total=int(zones_total),
            zone_cells_w=int(zone_cells_w),
            zone_cells_s=int(zone_cells_s),
        )
        space_case = _run_perf_case(
            label=f"space_gpu_z{zones_total}",
            requested_device="cuda",
            decomposition="3d",
            r0=r0,
            v0=v0,
            masses=masses,
            atom_types=atom_types,
            box=box,
            potential=potential,
            dt=float(dt),
            cutoff=float(cutoff),
            steps=int(steps),
            repeats=int(repeats),
            warmup=int(warmup),
            cell_size=float(cell_size),
            zones_total=int(zones_total),
            zone_cells_w=int(zone_cells_w),
            zone_cells_s=int(zone_cells_s),
            zones_nx=int(zones_nx),
            zones_ny=int(zones_ny),
            zones_nz=int(zones_nz),
        )
        time_case = _run_perf_case(
            label=f"time_gpu_z{zones_total}",
            requested_device="cuda",
            decomposition="1d",
            r0=r0,
            v0=v0,
            masses=masses,
            atom_types=atom_types,
            box=box,
            potential=potential,
            dt=float(dt),
            cutoff=float(cutoff),
            steps=int(steps),
            repeats=int(repeats),
            warmup=int(warmup),
            cell_size=float(cell_size),
            zones_total=int(zones_total),
            zone_cells_w=int(zone_cells_w),
            zone_cells_s=int(zone_cells_s),
            zones_nx=int(zones_nx),
            zones_ny=int(zones_ny),
            zones_nz=int(zones_nz),
        )
        td_speedup = None
        t_space = float(space_case.get("elapsed_sec_median", 0.0) or 0.0)
        t_time = float(time_case.get("elapsed_sec_median", 0.0) or 0.0)
        if t_space > 0.0 and t_time > 0.0:
            td_speedup = float(t_space / t_time)
        row_ok = bool(space_case.get("ok", False) and time_case.get("ok", False))
        gpu_effective_ok = (
            str(space_case.get("effective_device", "cpu")) == "cuda"
            and str(time_case.get("effective_device", "cpu")) == "cuda"
        )
        wave_batch_diag = dict(time_case.get("wave_batch_diagnostics", {}) or {})
        rows.append(
            {
                "zones_total": int(zones_total),
                "space_layout": f"{int(zones_nx)}x{int(zones_ny)}x{int(zones_nz)}",
                "ok": bool(row_ok),
                "gpu_effective_ok": bool(gpu_effective_ok),
                "space_gpu_median_sec": t_space,
                "time_gpu_median_sec": t_time,
                "space_gpu_steps_per_sec": float(
                    space_case.get("steps_per_sec_median", 0.0) or 0.0
                ),
                "time_gpu_steps_per_sec": float(time_case.get("steps_per_sec_median", 0.0) or 0.0),
                "td_speedup_vs_space": td_speedup,
                "space_gpu_effective_device": str(space_case.get("effective_device", "")),
                "time_gpu_effective_device": str(time_case.get("effective_device", "")),
                "space_gpu_error": str(space_case.get("error", "")),
                "time_gpu_error": str(time_case.get("error", "")),
                "wavefront_contract_version": str(wavefront.get("contract_version", "")),
                "wavefront_first_wave_size": int(wavefront.get("first_wave_size", 0) or 0),
                "wavefront_wave_size_max": int(wavefront.get("wave_size_max", 0) or 0),
                "wavefront_deferred_zones_total": int(
                    wavefront.get("deferred_zones_total", 0) or 0
                ),
                "wavefront_fallback_to_sequential_reasons": ",".join(
                    str(item) for item in wavefront.get("fallback_to_sequential_reasons", []) or []
                ),
                "time_gpu_wave_batch_enabled": bool(wave_batch_diag.get("enabled", False)),
                "time_gpu_wave_batch_launches_saved_per_step": float(
                    wave_batch_diag.get("launches_saved_per_step", 0.0) or 0.0
                ),
                "time_gpu_wave_batch_avg_successful_wave_size": float(
                    wave_batch_diag.get("avg_successful_wave_size", 0.0) or 0.0
                ),
                "time_gpu_wave_batch_neighbor_reuse_ratio": float(
                    wave_batch_diag.get("neighbor_reuse_ratio_weighted", 0.0) or 0.0
                ),
                "time_gpu_wave_batch_candidate_union_to_target_ratio": float(
                    wave_batch_diag.get("candidate_union_to_target_ratio_avg", 0.0) or 0.0
                ),
                "time_gpu_wave_batch_attempted_batches_per_step": float(
                    wave_batch_diag.get("attempted_wave_batches_per_step", 0.0) or 0.0
                ),
                "time_gpu_wave_batch_successful_batches_per_step": float(
                    wave_batch_diag.get("successful_wave_batches_per_step", 0.0) or 0.0
                ),
                "time_gpu_wave_batch_diagnostics_version": str(wave_batch_diag.get("version", "")),
                "time_gpu_wave_batch_runtime_contract_version": str(
                    wave_batch_diag.get("runtime_contract_version", "")
                ),
                "time_gpu_wave_batch": wave_batch_diag,
                "wavefront": wavefront,
                "space_gpu_case": dict(space_case),
                "time_gpu_case": dict(time_case),
            }
        )
    return rows


def _run_breakdown_for_layout(
    *,
    zones_total: int,
    zones_nx: int,
    zones_ny: int,
    zones_nz: int,
    r0,
    v0,
    masses,
    atom_types,
    box: float,
    potential,
    dt: float,
    cutoff: float,
    steps: int,
    repeats: int,
    warmup: int,
    cell_size: float,
    zone_cells_w: int,
    zone_cells_s: int,
) -> dict[str, object]:
    rows = []
    for case, decomposition in (("space_gpu", "3d"), ("time_gpu", "1d")):
        rows.append(
            _run_breakdown_case(
                label=case,
                decomposition=decomposition,
                requested_device="cuda",
                r0=r0,
                v0=v0,
                masses=masses,
                atom_types=atom_types,
                box=box,
                potential=potential,
                dt=float(dt),
                cutoff=float(cutoff),
                steps=int(steps),
                repeats=int(repeats),
                warmup=int(warmup),
                cell_size=float(cell_size),
                zones_total=int(zones_total),
                zone_cells_w=int(zone_cells_w),
                zone_cells_s=int(zone_cells_s),
                zones_nx=int(zones_nx),
                zones_ny=int(zones_ny),
                zones_nz=int(zones_nz),
            )
        )
    by_case = {str(row["case"]): dict(row) for row in rows}
    return {
        "ok_all": all(bool(row.get("ok", False)) for row in rows),
        "rows": rows,
        "by_case": by_case,
        "force_scope_contract": {
            "version": "pr_za01_v1",
            "baseline_reference_version": "pr_mb03_v1",
            "by_case": {
                case: {
                    "runtime_kind": str(
                        dict(row.get("breakdown", {}) or {}).get("many_body_runtime_kind", "")
                    ),
                    "evaluation_scope": str(
                        dict(row.get("breakdown", {}) or {}).get("many_body_evaluation_scope", "")
                    ),
                    "consumption_scope": str(
                        dict(row.get("breakdown", {}) or {}).get("many_body_consumption_scope", "")
                    ),
                    "target_local_available": int(
                        bool(
                            dict(row.get("breakdown", {}) or {}).get(
                                "many_body_target_local_available", 0
                            )
                        )
                    ),
                }
                for case, row in by_case.items()
            },
        },
    }


def _build_markdown_report(
    *,
    resources: dict[str, object],
    rows: list[dict[str, object]],
    recommendation: dict[str, object],
    breakdown: dict[str, object] | None,
    n_atoms: int,
    steps: int,
    repeats: int,
    warmup: int,
) -> str:
    lines = [
        "# TD Auto-Zoning Advisor",
        "",
        f"- n_atoms: `{int(n_atoms)}`",
        f"- steps: `{int(steps)}`",
        f"- repeats: `{int(repeats)}`",
        f"- warmup: `{int(warmup)}`",
        "",
        "## Resource Snapshot",
        "",
        f"- cpu_count: `{int(resources.get('cpu_count', 0) or 0)}`",
        f"- omp_threads_env: `{int(resources.get('omp_threads_env', 0) or 0)}`",
        f"- mpi_world_size_env: `{int(resources.get('mpi_world_size_env', 1) or 1)}`",
        f"- mpi_launchers: `{','.join(resources.get('mpi_launchers', []) or [])}`",
        f"- effective_gpu_backend: `{resources.get('effective_gpu_backend', 'cpu')}`",
        f"- gpu_count: `{int(resources.get('gpu_count', 0) or 0)}`",
        f"- gpu_names: `{','.join(resources.get('gpu_names', []) or [])}`",
        f"- resource_parallel_budget: `{int(resources.get('resource_parallel_budget', 0) or 0)}`",
        "",
        "## Candidate Layouts",
        "",
        "| zones_total | space_layout | space_gpu | time_gpu | td_speedup_vs_space | wavefront_first_wave | wavefront_deferred | wave_launches_saved/step | recommendation_rank |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    recommended_layout = str(
        recommendation.get("recommended_td_layout", {}).get("space_layout", "")
    )
    recommended_total = int(
        recommendation.get("recommended_td_layout", {}).get("zones_total", 0) or 0
    )
    for idx, row in enumerate(
        sorted(
            rows,
            key=lambda item: float(item.get("time_gpu_median_sec", float("inf")) or float("inf")),
        ),
        start=1,
    ):
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
            + str(int(row.get("wavefront_first_wave_size", 0) or 0))
            + " | "
            + str(int(row.get("wavefront_deferred_zones_total", 0) or 0))
            + " | "
            + f"{float(row.get('time_gpu_wave_batch_launches_saved_per_step', 0.0) or 0.0):.3f}"
            + " | "
            + (
                f"`#{idx}`"
                if int(row["zones_total"]) == recommended_total
                and str(row["space_layout"]) == recommended_layout
                else str(idx)
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- recommended_td_layout: zones_total=`{int(recommendation.get('recommended_td_layout', {}).get('zones_total', 0) or 0)}` layout=`{recommendation.get('recommended_td_layout', {}).get('space_layout', '')}`",
            f"- recommended_td_time_gpu: `{_fmt_time(float(recommendation.get('recommended_td_layout', {}).get('time_gpu_median_sec', 0.0) or 0.0))}`",
            f"- paired_td_speedup_vs_space: `{_fmt_speedup(recommendation.get('recommended_td_layout', {}).get('td_speedup_vs_space'))}`",
            f"- td_beats_paired_space: `{bool(recommendation.get('recommended_td_layout', {}).get('td_beats_paired_space', False))}`",
            f"- recommended_wavefront_first_wave: `{int(recommendation.get('recommended_td_layout', {}).get('wavefront_first_wave_size', 0) or 0)}`",
            f"- recommended_wavefront_deferred: `{int(recommendation.get('recommended_td_layout', {}).get('wavefront_deferred_zones_total', 0) or 0)}`",
            f"- recommended_wavefront_fallback: `{recommendation.get('recommended_td_layout', {}).get('wavefront_fallback_to_sequential_reasons', '')}`",
            f"- recommended_wave_batch_launches_saved_per_step: `{float(recommendation.get('recommended_td_layout', {}).get('time_gpu_wave_batch_launches_saved_per_step', 0.0) or 0.0):.3f}`",
            f"- recommended_wave_batch_neighbor_reuse: `{100.0 * float(recommendation.get('recommended_td_layout', {}).get('time_gpu_wave_batch_neighbor_reuse_ratio', 0.0) or 0.0):.2f}%`",
            f"- recommended_wave_batch_union_to_target_ratio: `{float(recommendation.get('recommended_td_layout', {}).get('time_gpu_wave_batch_candidate_union_to_target_ratio', 0.0) or 0.0):.3f}`",
            f"- best_overall_case: `{recommendation.get('best_overall_case', {}).get('case', '')}`",
            f"- best_overall_layout: zones_total=`{int(recommendation.get('best_overall_case', {}).get('zones_total', 0) or 0)}` layout=`{recommendation.get('best_overall_case', {}).get('space_layout', '')}`",
            f"- advisory_status: `{recommendation.get('advisory_status', '')}`",
            "",
        ]
    )
    if breakdown is not None:
        time_gpu = dict(dict(breakdown.get("by_case", {}) or {}).get("time_gpu", {}) or {})
        time_breakdown = dict(time_gpu.get("breakdown", {}) or {})
        lines.extend(
            [
                "## Breakdown Evidence",
                "",
                f"- current_contract_version: `{dict(breakdown.get('force_scope_contract', {}) or {}).get('version', '')}`",
                f"- baseline_reference_version: `{dict(breakdown.get('force_scope_contract', {}) or {}).get('baseline_reference_version', '')}`",
                f"- time_gpu_evaluation_scope: `{time_breakdown.get('many_body_evaluation_scope', '')}`",
                f"- time_gpu_forces_full_share: `{100.0 * float(time_breakdown.get('forces_full_share', 0.0) or 0.0):.2f}%`",
                f"- time_gpu_target_local_force_calls: `{int(round(float(time_breakdown.get('target_local_force_calls', 0.0) or 0.0)))}`",
                f"- time_gpu_wave_batch_launches_saved_per_step: `{float(dict(time_breakdown.get('wave_batch_diagnostics', {}) or {}).get('launches_saved_per_step', 0.0) or 0.0):.3f}`",
                f"- time_gpu_wave_batch_neighbor_reuse: `{100.0 * float(dict(time_breakdown.get('wave_batch_diagnostics', {}) or {}).get('neighbor_reuse_ratio_weighted', 0.0) or 0.0):.2f}%`",
                f"- time_gpu_wave_batch_union_to_target_ratio: `{float(dict(time_breakdown.get('wave_batch_diagnostics', {}) or {}).get('candidate_union_to_target_ratio_avg', 0.0) or 0.0):.3f}`",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _build_recommendation(
    rows: list[dict[str, object]], resources: dict[str, object]
) -> dict[str, object]:
    def _td_sort_key(row: dict[str, object]) -> tuple[float, float, float, float, int, int]:
        pressure = float(
            row.get("time_gpu_wave_batch_candidate_union_to_target_ratio", float("inf"))
            or float("inf")
        )
        return (
            float(row.get("time_gpu_median_sec", float("inf")) or float("inf")),
            -float(row.get("time_gpu_wave_batch_launches_saved_per_step", 0.0) or 0.0),
            -float(row.get("time_gpu_wave_batch_neighbor_reuse_ratio", 0.0) or 0.0),
            pressure,
            int(row.get("wavefront_deferred_zones_total", 0) or 0),
            -int(row.get("wavefront_first_wave_size", 0) or 0),
        )

    ok_rows = [row for row in rows if bool(row.get("ok", False))]
    if not ok_rows:
        return {
            "advisory_status": "no_valid_layouts",
            "recommended_td_layout": None,
            "best_overall_case": None,
            "resource_parallel_budget": int(resources.get("resource_parallel_budget", 0) or 0),
        }

    best_td = min(ok_rows, key=_td_sort_key)
    best_overall_candidates: list[tuple[str, float, dict[str, object]]] = []
    for row in ok_rows:
        best_overall_candidates.append(
            ("space_gpu", float(row.get("space_gpu_median_sec", 0.0) or 0.0), row)
        )
        best_overall_candidates.append(
            ("time_gpu", float(row.get("time_gpu_median_sec", 0.0) or 0.0), row)
        )
    best_overall_case, _best_overall_time, best_overall_row = min(
        best_overall_candidates, key=lambda item: item[1]
    )

    td_speedup = best_td.get("td_speedup_vs_space")
    td_beats = bool(td_speedup is not None and float(td_speedup) > 1.0)
    wave_saved = float(best_td.get("time_gpu_wave_batch_launches_saved_per_step", 0.0) or 0.0)
    wave_enabled = bool(best_td.get("time_gpu_wave_batch_enabled", False))
    wave_reuse = float(best_td.get("time_gpu_wave_batch_neighbor_reuse_ratio", 0.0) or 0.0)
    if td_beats and wave_enabled and wave_saved > 0.0:
        advisory_status = "td_favorable"
    elif wave_enabled and wave_saved > 0.0:
        advisory_status = "td_observed_with_runtime_wavefront_but_not_faster_than_space"
    elif wave_enabled:
        advisory_status = "td_observed_with_weak_runtime_wavefront_signal"
    else:
        advisory_status = "td_observed_without_runtime_wavefront_realization"
    return {
        "advisory_status": advisory_status,
        "resource_parallel_budget": int(resources.get("resource_parallel_budget", 0) or 0),
        "recommended_td_layout": {
            "zones_total": int(best_td["zones_total"]),
            "space_layout": str(best_td["space_layout"]),
            "time_gpu_median_sec": float(best_td.get("time_gpu_median_sec", 0.0) or 0.0),
            "space_gpu_median_sec": float(best_td.get("space_gpu_median_sec", 0.0) or 0.0),
            "td_speedup_vs_space": td_speedup,
            "td_beats_paired_space": td_beats,
            "wavefront_first_wave_size": int(best_td.get("wavefront_first_wave_size", 0) or 0),
            "wavefront_deferred_zones_total": int(
                best_td.get("wavefront_deferred_zones_total", 0) or 0
            ),
            "wavefront_fallback_to_sequential_reasons": str(
                best_td.get("wavefront_fallback_to_sequential_reasons", "")
            ),
            "time_gpu_wave_batch_enabled": bool(best_td.get("time_gpu_wave_batch_enabled", False)),
            "time_gpu_wave_batch_launches_saved_per_step": float(
                best_td.get("time_gpu_wave_batch_launches_saved_per_step", 0.0) or 0.0
            ),
            "time_gpu_wave_batch_avg_successful_wave_size": float(
                best_td.get("time_gpu_wave_batch_avg_successful_wave_size", 0.0) or 0.0
            ),
            "time_gpu_wave_batch_neighbor_reuse_ratio": float(wave_reuse),
            "time_gpu_wave_batch_candidate_union_to_target_ratio": float(
                best_td.get("time_gpu_wave_batch_candidate_union_to_target_ratio", 0.0) or 0.0
            ),
            "time_gpu_wave_batch_attempted_batches_per_step": float(
                best_td.get("time_gpu_wave_batch_attempted_batches_per_step", 0.0) or 0.0
            ),
            "time_gpu_wave_batch_successful_batches_per_step": float(
                best_td.get("time_gpu_wave_batch_successful_batches_per_step", 0.0) or 0.0
            ),
        },
        "best_overall_case": {
            "case": best_overall_case,
            "zones_total": int(best_overall_row["zones_total"]),
            "space_layout": str(best_overall_row["space_layout"]),
            "median_sec": float(
                best_overall_row.get(f"{best_overall_case}_median_sec", 0.0)
                or best_overall_row.get("time_gpu_median_sec", 0.0)
                or 0.0
            ),
        },
        "cost_model_sort_key": {
            "time_gpu_median_sec": float(best_td.get("time_gpu_median_sec", 0.0) or 0.0),
            "wave_batch_launches_saved_per_step": float(wave_saved),
            "wave_batch_neighbor_reuse_ratio": float(wave_reuse),
            "wave_batch_candidate_union_to_target_ratio": float(
                best_td.get("time_gpu_wave_batch_candidate_union_to_target_ratio", 0.0) or 0.0
            ),
            "wavefront_deferred_zones_total": int(
                best_td.get("wavefront_deferred_zones_total", 0) or 0
            ),
            "wavefront_first_wave_size": int(best_td.get("wavefront_first_wave_size", 0) or 0),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Resource-aware TD auto-zoning advisor for EAM/eam-alloy without changing runtime policy"
    )
    ap.add_argument("--out", default="results/eam_autozoning_advisor.csv")
    ap.add_argument("--md", default="results/eam_autozoning_advisor.md")
    ap.add_argument("--json", default="results/eam_autozoning_advisor.summary.json")
    ap.add_argument("--n-atoms", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=64)
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
        "--zone-totals",
        default="",
        help="optional comma-separated zone totals override; when omitted, derive candidates from detected resources",
    )
    ap.add_argument("--max-zones-total", type=int, default=0)
    ap.add_argument(
        "--eam-file",
        default="examples/potentials/eam_alloy/AlCu.eam.alloy",
        help="EAM/alloy setfl file used for the benchmark potential",
    )
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--require-effective-cuda", action="store_true")
    ap.add_argument("--skip-breakdown", action="store_true")
    args = ap.parse_args()

    resources = _detect_resources()
    if str(args.zone_totals).strip():
        zone_totals = _parse_zone_totals(str(args.zone_totals))
    else:
        parallel_budget = int(resources.get("resource_parallel_budget", 2) or 2)
        max_zones_total = (
            int(args.max_zones_total) if int(args.max_zones_total) > 0 else parallel_budget
        )
        zone_totals = [
            value for value in _default_zone_totals(parallel_budget) if value <= max_zones_total
        ]
        if not zone_totals:
            zone_totals = [2]

    r0, v0, masses, atom_types, box = _build_alloy_state(
        n_atoms=int(args.n_atoms),
        lattice_a=float(args.lattice_a),
        jitter=float(args.jitter),
        seed=int(args.seed),
        velocity_std=float(args.velocity_std),
    )
    layouts = _enumerate_candidate_layouts(
        box=float(box),
        cutoff=float(args.cutoff),
        cell_size=float(args.cell_size),
        zone_cells_w=int(args.zone_cells_w),
        zone_cells_s=int(args.zone_cells_s),
        zone_totals=zone_totals,
    )
    resources["selected_zone_totals"] = [int(item[0]) for item in layouts]
    resources["selected_layouts"] = [f"{z}:{nx}x{ny}x{nz}" for z, nx, ny, nz in layouts]

    potential = make_potential(
        "eam/alloy",
        {"file": str(args.eam_file), "elements": ["Al", "Cu"]},
    )
    rows = _layout_summary_rows(
        layouts=layouts,
        r0=r0,
        v0=v0,
        masses=masses,
        atom_types=atom_types,
        box=float(box),
        potential=potential,
        dt=float(args.dt),
        cutoff=float(args.cutoff),
        steps=int(args.steps),
        repeats=int(args.repeats),
        warmup=int(args.warmup),
        cell_size=float(args.cell_size),
        zone_cells_w=int(args.zone_cells_w),
        zone_cells_s=int(args.zone_cells_s),
    )
    if bool(args.require_effective_cuda):
        for row in rows:
            row["ok"] = bool(row.get("ok", False) and row.get("gpu_effective_ok", False))

    recommendation = _build_recommendation(rows, resources)
    breakdown = None
    if not bool(args.skip_breakdown) and recommendation.get("recommended_td_layout"):
        best = dict(recommendation["recommended_td_layout"])
        total = int(best["zones_total"])
        nx, ny, nz = [int(v) for v in str(best["space_layout"]).split("x")]
        breakdown = _run_breakdown_for_layout(
            zones_total=total,
            zones_nx=nx,
            zones_ny=ny,
            zones_nz=nz,
            r0=r0,
            v0=v0,
            masses=masses,
            atom_types=atom_types,
            box=float(box),
            potential=potential,
            dt=float(args.dt),
            cutoff=float(args.cutoff),
            steps=int(args.steps),
            repeats=int(args.repeats),
            warmup=int(args.warmup),
            cell_size=float(args.cell_size),
            zone_cells_w=int(args.zone_cells_w),
            zone_cells_s=int(args.zone_cells_s),
        )
        if bool(args.require_effective_cuda):
            breakdown["ok_all"] = bool(
                breakdown.get("ok_all", False)
                and all(
                    str(row.get("effective_device", "cpu")) == "cuda"
                    for row in breakdown.get("rows", [])
                )
            )

    report_markdown = _build_markdown_report(
        resources=resources,
        rows=rows,
        recommendation=recommendation,
        breakdown=breakdown,
        n_atoms=int(args.n_atoms),
        steps=int(args.steps),
        repeats=int(args.repeats),
        warmup=int(args.warmup),
    )

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
                "space_gpu_median_sec",
                "time_gpu_median_sec",
                "td_speedup_vs_space",
                "wavefront_contract_version",
                "wavefront_first_wave_size",
                "wavefront_wave_size_max",
                "wavefront_deferred_zones_total",
                "wavefront_fallback_to_sequential_reasons",
                "time_gpu_wave_batch_enabled",
                "time_gpu_wave_batch_launches_saved_per_step",
                "time_gpu_wave_batch_neighbor_reuse_ratio",
                "time_gpu_wave_batch_candidate_union_to_target_ratio",
                "space_gpu_effective_device",
                "time_gpu_effective_device",
            ]
        )
        for row in rows:
            w.writerow(
                [
                    int(row["zones_total"]),
                    row["space_layout"],
                    int(bool(row.get("ok", False))),
                    int(bool(row.get("gpu_effective_ok", False))),
                    f"{float(row.get('space_gpu_median_sec', 0.0) or 0.0):.6f}",
                    f"{float(row.get('time_gpu_median_sec', 0.0) or 0.0):.6f}",
                    (
                        f"{float(row['td_speedup_vs_space']):.6f}"
                        if row.get("td_speedup_vs_space") is not None
                        else ""
                    ),
                    row.get("wavefront_contract_version", ""),
                    int(row.get("wavefront_first_wave_size", 0) or 0),
                    int(row.get("wavefront_wave_size_max", 0) or 0),
                    int(row.get("wavefront_deferred_zones_total", 0) or 0),
                    row.get("wavefront_fallback_to_sequential_reasons", ""),
                    int(bool(row.get("time_gpu_wave_batch_enabled", False))),
                    f"{float(row.get('time_gpu_wave_batch_launches_saved_per_step', 0.0) or 0.0):.6f}",
                    f"{float(row.get('time_gpu_wave_batch_neighbor_reuse_ratio', 0.0) or 0.0):.6f}",
                    f"{float(row.get('time_gpu_wave_batch_candidate_union_to_target_ratio', 0.0) or 0.0):.6f}",
                    row.get("space_gpu_effective_device", ""),
                    row.get("time_gpu_effective_device", ""),
                ]
            )

    out_md.write_text(report_markdown, encoding="utf-8")

    ok_all = all(bool(row.get("ok", False)) for row in rows)
    if breakdown is not None:
        ok_all = bool(ok_all and bool(breakdown.get("ok_all", False)))

    summary = {
        "n": len(rows),
        "total": len(rows),
        "ok": int(sum(1 for row in rows if bool(row.get("ok", False)))),
        "fail": int(sum(1 for row in rows if not bool(row.get("ok", False)))),
        "ok_all": bool(ok_all),
        "resources": resources,
        "rows": rows,
        "by_layout": {str(row["space_layout"]): dict(row) for row in rows},
        "wavefront_contract_version": (
            str(rows[0].get("wavefront_contract_version", "")) if rows else ""
        ),
        "wavefront_by_zones_total": {
            str(int(row["zones_total"])): dict(row.get("wavefront", {}) or {}) for row in rows
        },
        "recommendation": recommendation,
        "breakdown": breakdown,
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
    print(f"[td-autozoning-advisor] wrote {out_csv}")
    print(f"[td-autozoning-advisor] wrote {out_md}")
    print(f"[td-autozoning-advisor] wrote {out_json}")
    if bool(args.strict) and not bool(summary["ok_all"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
