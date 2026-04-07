#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np

from tdmd.backend import resolve_backend
from tdmd.forces_gpu import get_last_device_state_sync_diagnostics, reset_device_state_cache
from tdmd.potentials import make_potential
import tdmd.forces_gpu as forces_gpu
import tdmd.td_local as td_local

from bench_eam_decomp_perf import _build_alloy_state


@dataclass
class _SectionStat:
    calls: int = 0
    inclusive_sec: float = 0.0
    exclusive_sec: float = 0.0
    synced_atoms: int = 0
    full_sync_calls: int = 0
    dirty_tracking_calls: int = 0


class _NestedTimer:
    def __init__(self) -> None:
        self.stats: dict[str, _SectionStat] = {}
        self._stack: list[dict[str, float | str]] = []

    @contextmanager
    def section(self, name: str):
        frame: dict[str, float | str] = {"name": str(name), "child_sec": 0.0}
        self._stack.append(frame)
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = float(time.perf_counter() - started)
            self._stack.pop()
            stat = self.stats.setdefault(str(name), _SectionStat())
            stat.calls += 1
            stat.inclusive_sec += elapsed
            child_sec = float(frame["child_sec"])
            stat.exclusive_sec += max(0.0, elapsed - child_sec)
            if self._stack:
                self._stack[-1]["child_sec"] = float(self._stack[-1]["child_sec"]) + elapsed

    def add(self, name: str, **increments: int) -> None:
        stat = self.stats.setdefault(str(name), _SectionStat())
        for key, value in increments.items():
            setattr(stat, key, int(getattr(stat, key)) + int(value))

    def stat(self, name: str) -> _SectionStat:
        return self.stats.get(str(name), _SectionStat())


def _fmt_time(value: float) -> str:
    return f"{float(value):.6f}s" if float(value) > 0.0 else "n/a"


def _fmt_rate(value: float) -> str:
    return f"{float(value):.3f}" if float(value) > 0.0 else "n/a"


def _fmt_share(value: float) -> str:
    return f"{100.0 * float(value):.2f}%" if float(value) > 0.0 else "0.00%"


def _fmt_speedup(value: float | None) -> str:
    return f"{float(value):.3f}x" if value is not None else "n/a"


def _speedup(space_row: dict[str, object], time_row: dict[str, object]) -> float | None:
    t_space = float(space_row.get("elapsed_sec_median", 0.0) or 0.0)
    t_time = float(time_row.get("elapsed_sec_median", 0.0) or 0.0)
    if t_space <= 0.0 or t_time <= 0.0:
        return None
    return float(t_space / t_time)


def _profiled_run(
    *,
    decomposition: str,
    requested_device: str,
    r0,
    v0,
    masses,
    atom_types,
    box: float,
    potential,
    dt: float,
    cutoff: float,
    steps: int,
    cell_size: float,
    zones_total: int,
    zone_cells_w: int,
    zone_cells_s: int,
    zones_nx: int,
    zones_ny: int,
    zones_nz: int,
) -> tuple[float, dict[str, object]]:
    timer = _NestedTimer()
    reset_device_state_cache()
    effective_device = str(getattr(resolve_backend(str(requested_device)), "device", "cpu"))
    many_body_scope = td_local.describe_many_body_force_scope(
        potential,
        sync_mode=False,
        decomposition=decomposition,
        device=effective_device,
    ) or {}

    orig_forces_full = td_local._TDLocalCtx.forces_full
    orig_build_cell_list = td_local.build_cell_list
    orig_zone_assign_1d = td_local.assign_atoms_to_zones
    orig_zone_assign_3d = td_local.assign_atoms_to_zones_3d
    orig_overlap_1d = td_local.zones_overlapping_range_pbc
    orig_overlap_3d = td_local.zones_overlapping_aabb_pbc
    orig_buffer_skin = td_local.compute_zone_buffer_skin
    orig_get_device_state = forces_gpu._get_device_state
    orig_try_gpu_forces = td_local.try_gpu_forces_on_targets
    target_local_owner = type(potential)
    orig_target_local_force = getattr(target_local_owner, "forces_on_targets", None)

    def wrap_with_section(name: str, fn):
        def _wrapped(*args, **kwargs):
            with timer.section(name):
                return fn(*args, **kwargs)

        return _wrapped

    def wrapped_get_device_state(*args, **kwargs):
        with timer.section("device_sync"):
            out = orig_get_device_state(*args, **kwargs)
        diag = get_last_device_state_sync_diagnostics()
        timer.add(
            "device_sync",
            synced_atoms=int(diag.last_synced_atoms),
            full_sync_calls=int(bool(diag.full_sync)),
            dirty_tracking_calls=int(bool(diag.used_dirty_tracking)),
        )
        return out

    def wrapped_target_local_force(self, *args, **kwargs):
        with timer.section("target_local_force"):
            return orig_target_local_force(self, *args, **kwargs)

    def wrapped_try_gpu_forces_on_targets(*args, **kwargs):
        target_ids = np.asarray(kwargs.get("target_ids"), dtype=np.int32)
        candidate_ids = np.asarray(kwargs.get("candidate_ids"), dtype=np.int32)
        n_atoms = int(np.asarray(kwargs.get("r")).shape[0])
        full_targets = bool(target_ids.size == n_atoms and np.array_equal(target_ids, np.arange(n_atoms, dtype=np.int32)))
        full_candidates = bool(candidate_ids.size == n_atoms and np.array_equal(candidate_ids, np.arange(n_atoms, dtype=np.int32)))
        if full_targets and full_candidates:
            return orig_try_gpu_forces(*args, **kwargs)
        with timer.section("target_local_force"):
            return orig_try_gpu_forces(*args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                td_local._TDLocalCtx,
                "forces_full",
                new=wrap_with_section("forces_full", orig_forces_full),
            )
        )
        stack.enter_context(
            mock.patch.object(
                td_local,
                "build_cell_list",
                new=wrap_with_section("build_cell_list", orig_build_cell_list),
            )
        )
        stack.enter_context(
            mock.patch.object(
                td_local,
                "assign_atoms_to_zones",
                new=wrap_with_section("zone_assign_1d", orig_zone_assign_1d),
            )
        )
        stack.enter_context(
            mock.patch.object(
                td_local,
                "assign_atoms_to_zones_3d",
                new=wrap_with_section("zone_assign_3d", orig_zone_assign_3d),
            )
        )
        stack.enter_context(
            mock.patch.object(
                td_local,
                "zones_overlapping_range_pbc",
                new=wrap_with_section("candidate_enum_1d", orig_overlap_1d),
            )
        )
        stack.enter_context(
            mock.patch.object(
                td_local,
                "zones_overlapping_aabb_pbc",
                new=wrap_with_section("candidate_enum_3d", orig_overlap_3d),
            )
        )
        stack.enter_context(
            mock.patch.object(
                td_local,
                "compute_zone_buffer_skin",
                new=wrap_with_section("zone_buffer_skin", orig_buffer_skin),
            )
        )
        stack.enter_context(mock.patch.object(forces_gpu, "_get_device_state", new=wrapped_get_device_state))
        stack.enter_context(
            mock.patch.object(td_local, "try_gpu_forces_on_targets", new=wrapped_try_gpu_forces_on_targets)
        )
        if orig_target_local_force is not None:
            stack.enter_context(
                mock.patch.object(target_local_owner, "forces_on_targets", new=wrapped_target_local_force)
            )
        r = r0.copy()
        v = v0.copy()
        started = time.perf_counter()
        run_td = td_local.run_td_local
        run_td(
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
        elapsed = float(time.perf_counter() - started)

    forces_full = timer.stat("forces_full")
    target_local_force = timer.stat("target_local_force")
    device_sync = timer.stat("device_sync")
    build_cell_list = timer.stat("build_cell_list")
    zone_buffer_skin = timer.stat("zone_buffer_skin")
    candidate_enum_1d = timer.stat("candidate_enum_1d")
    candidate_enum_3d = timer.stat("candidate_enum_3d")
    zone_assign_1d = timer.stat("zone_assign_1d")
    zone_assign_3d = timer.stat("zone_assign_3d")

    candidate_enum_sec = candidate_enum_1d.inclusive_sec + candidate_enum_3d.inclusive_sec
    candidate_enum_calls = candidate_enum_1d.calls + candidate_enum_3d.calls
    zone_assign_sec = zone_assign_1d.inclusive_sec + zone_assign_3d.inclusive_sec
    zone_assign_calls = zone_assign_1d.calls + zone_assign_3d.calls
    forces_full_compute_self_sec = max(0.0, forces_full.exclusive_sec)
    target_local_force_compute_self_sec = max(0.0, target_local_force.exclusive_sec)
    other_sec = max(
        0.0,
        elapsed
        - (
            forces_full_compute_self_sec
            + target_local_force_compute_self_sec
            + device_sync.inclusive_sec
            + build_cell_list.inclusive_sec
            + zone_buffer_skin.inclusive_sec
            + candidate_enum_sec
            + zone_assign_sec
        ),
    )
    fallback_from_cuda = bool(str(requested_device) == "cuda" and effective_device != "cuda")
    wave_batch_diagnostics = td_local.get_last_td_local_wave_batch_diagnostics().as_dict()

    breakdown = {
        "requested_device": str(requested_device),
        "effective_device": effective_device,
        "fallback_from_cuda": int(fallback_from_cuda),
        "elapsed_sec": float(elapsed),
        "steps_per_sec": (float(steps) / float(elapsed)) if float(elapsed) > 0.0 else 0.0,
        "many_body_runtime_kind": str(many_body_scope.get("runtime_kind", "")),
        "many_body_evaluation_scope": str(many_body_scope.get("evaluation_scope", "")),
        "many_body_consumption_scope": str(many_body_scope.get("consumption_scope", "")),
        "many_body_target_local_available": int(bool(many_body_scope.get("target_local_available", False))),
        "many_body_scope_rationale": str(many_body_scope.get("rationale", "")),
        "forces_full_total_sec": float(forces_full.inclusive_sec),
        "forces_full_compute_self_sec": float(forces_full_compute_self_sec),
        "forces_full_calls": int(forces_full.calls),
        "forces_full_calls_per_step": (float(forces_full.calls) / float(steps)) if int(steps) > 0 else 0.0,
        "target_local_force_total_sec": float(target_local_force.inclusive_sec),
        "target_local_force_compute_self_sec": float(target_local_force_compute_self_sec),
        "target_local_force_calls": int(target_local_force.calls),
        "target_local_force_calls_per_step": (
            float(target_local_force.calls) / float(steps) if int(steps) > 0 else 0.0
        ),
        "device_sync_sec": float(device_sync.inclusive_sec),
        "device_sync_calls": int(device_sync.calls),
        "device_sync_atoms_total": int(device_sync.synced_atoms),
        "device_sync_full_calls": int(device_sync.full_sync_calls),
        "device_sync_dirty_tracking_calls": int(device_sync.dirty_tracking_calls),
        "avg_synced_atoms_per_call": (
            float(device_sync.synced_atoms) / float(device_sync.calls) if int(device_sync.calls) > 0 else 0.0
        ),
        "build_cell_list_sec": float(build_cell_list.inclusive_sec),
        "build_cell_list_calls": int(build_cell_list.calls),
        "zone_buffer_skin_sec": float(zone_buffer_skin.inclusive_sec),
        "zone_buffer_skin_calls": int(zone_buffer_skin.calls),
        "candidate_enum_sec": float(candidate_enum_sec),
        "candidate_enum_calls": int(candidate_enum_calls),
        "zone_assign_sec": float(zone_assign_sec),
        "zone_assign_calls": int(zone_assign_calls),
        "other_sec": float(other_sec),
        "forces_full_share": (
            float(forces_full.inclusive_sec) / float(elapsed) if float(elapsed) > 0.0 else 0.0
        ),
        "target_local_force_share": (
            float(target_local_force.inclusive_sec) / float(elapsed) if float(elapsed) > 0.0 else 0.0
        ),
        "device_sync_share": (
            float(device_sync.inclusive_sec) / float(elapsed) if float(elapsed) > 0.0 else 0.0
        ),
        "build_cell_list_share": (
            float(build_cell_list.inclusive_sec) / float(elapsed) if float(elapsed) > 0.0 else 0.0
        ),
        "zone_buffer_skin_share": (
            float(zone_buffer_skin.inclusive_sec) / float(elapsed) if float(elapsed) > 0.0 else 0.0
        ),
        "candidate_enum_share": float(candidate_enum_sec) / float(elapsed) if float(elapsed) > 0.0 else 0.0,
        "zone_assign_share": float(zone_assign_sec) / float(elapsed) if float(elapsed) > 0.0 else 0.0,
        "other_share": float(other_sec) / float(elapsed) if float(elapsed) > 0.0 else 0.0,
        "wave_batch_diagnostics": wave_batch_diagnostics,
    }
    return elapsed, breakdown


def _run_case(
    *,
    label: str,
    decomposition: str,
    requested_device: str,
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
    zones_total: int,
    zone_cells_w: int,
    zone_cells_s: int,
    zones_nx: int,
    zones_ny: int,
    zones_nz: int,
) -> dict[str, object]:
    elapsed_runs: list[float] = []
    breakdown_runs: list[dict[str, object]] = []
    error = ""

    try:
        for _ in range(max(0, int(warmup))):
            _profiled_run(
                decomposition=decomposition,
                requested_device=requested_device,
                r0=r0,
                v0=v0,
                masses=masses,
                atom_types=atom_types,
                box=box,
                potential=potential,
                dt=dt,
                cutoff=cutoff,
                steps=steps,
                cell_size=cell_size,
                zones_total=zones_total,
                zone_cells_w=zone_cells_w,
                zone_cells_s=zone_cells_s,
                zones_nx=zones_nx,
                zones_ny=zones_ny,
                zones_nz=zones_nz,
            )
        for _ in range(max(1, int(repeats))):
            elapsed, breakdown = _profiled_run(
                decomposition=decomposition,
                requested_device=requested_device,
                r0=r0,
                v0=v0,
                masses=masses,
                atom_types=atom_types,
                box=box,
                potential=potential,
                dt=dt,
                cutoff=cutoff,
                steps=steps,
                cell_size=cell_size,
                zones_total=zones_total,
                zone_cells_w=zone_cells_w,
                zone_cells_s=zone_cells_s,
                zones_nx=zones_nx,
                zones_ny=zones_ny,
                zones_nz=zones_nz,
            )
            elapsed_runs.append(float(elapsed))
            breakdown_runs.append(dict(breakdown))
    except Exception as exc:
        error = str(exc)

    ok = bool(breakdown_runs) and not error
    median_elapsed = float(sorted(elapsed_runs)[len(elapsed_runs) // 2]) if elapsed_runs else 0.0

    aggregate_keys = [
        "elapsed_sec",
        "steps_per_sec",
        "many_body_target_local_available",
        "forces_full_total_sec",
        "forces_full_compute_self_sec",
        "forces_full_calls",
        "forces_full_calls_per_step",
        "target_local_force_total_sec",
        "target_local_force_compute_self_sec",
        "target_local_force_calls",
        "target_local_force_calls_per_step",
        "device_sync_sec",
        "device_sync_calls",
        "device_sync_atoms_total",
        "device_sync_full_calls",
        "device_sync_dirty_tracking_calls",
        "avg_synced_atoms_per_call",
        "build_cell_list_sec",
        "build_cell_list_calls",
        "zone_buffer_skin_sec",
        "zone_buffer_skin_calls",
        "candidate_enum_sec",
        "candidate_enum_calls",
        "zone_assign_sec",
        "zone_assign_calls",
        "other_sec",
        "forces_full_share",
        "target_local_force_share",
        "device_sync_share",
        "build_cell_list_share",
        "zone_buffer_skin_share",
        "candidate_enum_share",
        "zone_assign_share",
        "other_share",
    ]
    aggregated: dict[str, object] = {}
    for key in aggregate_keys:
        values = [float(run.get(key, 0.0) or 0.0) for run in breakdown_runs]
        aggregated[key] = (sum(values) / float(len(values))) if values else 0.0

    if breakdown_runs:
        last = breakdown_runs[-1]
        aggregated["requested_device"] = str(last.get("requested_device", requested_device))
        aggregated["effective_device"] = str(last.get("effective_device", "cpu"))
        aggregated["fallback_from_cuda"] = int(last.get("fallback_from_cuda", 0))
        aggregated["many_body_runtime_kind"] = str(last.get("many_body_runtime_kind", ""))
        aggregated["many_body_evaluation_scope"] = str(last.get("many_body_evaluation_scope", ""))
        aggregated["many_body_consumption_scope"] = str(last.get("many_body_consumption_scope", ""))
        aggregated["many_body_scope_rationale"] = str(last.get("many_body_scope_rationale", ""))
        aggregated["wave_batch_diagnostics"] = td_local.aggregate_td_local_wave_batch_diagnostics(
            [dict(run.get("wave_batch_diagnostics", {}) or {}) for run in breakdown_runs]
        )
    else:
        effective = str(getattr(resolve_backend(str(requested_device)), "device", "cpu"))
        aggregated["requested_device"] = str(requested_device)
        aggregated["effective_device"] = effective
        aggregated["fallback_from_cuda"] = int(str(requested_device) == "cuda" and effective != "cuda")
        aggregated["many_body_runtime_kind"] = ""
        aggregated["many_body_evaluation_scope"] = ""
        aggregated["many_body_consumption_scope"] = ""
        aggregated["many_body_scope_rationale"] = ""
        aggregated["wave_batch_diagnostics"] = td_local.aggregate_td_local_wave_batch_diagnostics([])

    return {
        "case": str(label),
        "decomposition_kind": ("time" if str(decomposition) == "1d" else "space"),
        "decomposition": str(decomposition),
        "requested_device": str(aggregated.get("requested_device", requested_device)),
        "effective_device": str(aggregated.get("effective_device", "cpu")),
        "fallback_from_cuda": int(aggregated.get("fallback_from_cuda", 0)),
        "repeats": int(repeats),
        "warmup": int(warmup),
        "ok": bool(ok),
        "error": error,
        "elapsed_sec_median": float(median_elapsed),
        "steps_per_sec_median": (
            float(steps) / float(median_elapsed) if float(median_elapsed) > 0.0 else 0.0
        ),
        "breakdown": aggregated,
        "breakdown_runs": breakdown_runs,
    }


def _build_report(
    *,
    rows_by_case: dict[str, dict[str, object]],
    n_atoms: int,
    steps: int,
    repeats: int,
    warmup: int,
    zones_total: int,
    zones_nx: int,
    zones_ny: int,
    zones_nz: int,
) -> str:
    col_order = ["space_gpu", "time_gpu"]

    def _row(metric: str, formatter) -> str:
        return "| " + metric + " | " + " | ".join(formatter(case) for case in col_order) + " |"

    def _b(case: str, key: str) -> float:
        return float(dict(rows_by_case[case].get("breakdown", {}) or {}).get(key, 0.0) or 0.0)

    def _s(case: str, key: str) -> str:
        return str(dict(rows_by_case[case].get("breakdown", {}) or {}).get(key, "") or "")

    lines = [
        "# EAM TD GPU Breakdown",
        "",
        f"- n_atoms: `{int(n_atoms)}`",
        f"- steps: `{int(steps)}`",
        f"- repeats: `{int(repeats)}`",
        f"- warmup: `{int(warmup)}`",
        f"- zones_total: `{int(zones_total)}`",
        f"- space_layout: `{int(zones_nx)}x{int(zones_ny)}x{int(zones_nz)}`",
        "- interpretation: `space_gpu` uses `decomposition=3d`; `time_gpu` uses `decomposition=1d` with the same total zone count.",
        "- `force_scope_contract.version` reports the current many-body td_local contract, while `baseline_reference_version` points to the frozen pre-locality baseline (`pr_mb01_v1`). `evaluation_scope` tells where forces are evaluated, and `consumption_scope` tells whether td_local uses the full result or slices it to target ids.",
        "- `wave_batch_diagnostics.version` reports the current runtime observability contract for fused pre-force slab waves (`pr_sw05_v1`).",
        "- timing rows below are exclusive where noted: `forces_full_compute_self_sec` excludes nested device-sync time.",
        "",
        "| metric | space_gpu | time_gpu |",
        "|---|---:|---:|",
        _row("effective_device", lambda case: f"`{rows_by_case[case].get('effective_device', 'cpu')}`"),
        _row("fallback_from_cuda", lambda case: str(int(rows_by_case[case].get("fallback_from_cuda", 0)))),
        _row("many_body_eval_scope", lambda case: f"`{_s(case, 'many_body_evaluation_scope') or 'n/a'}`"),
        _row("many_body_consumption_scope", lambda case: f"`{_s(case, 'many_body_consumption_scope') or 'n/a'}`"),
        _row("many_body_target_local_available", lambda case: str(int(round(_b(case, "many_body_target_local_available"))))),
        _row("median_sec", lambda case: _fmt_time(float(rows_by_case[case].get("elapsed_sec_median", 0.0) or 0.0))),
        _row("steps_per_sec", lambda case: _fmt_rate(float(rows_by_case[case].get("steps_per_sec_median", 0.0) or 0.0))),
        _row("forces_full_total_sec", lambda case: _fmt_time(_b(case, "forces_full_total_sec"))),
        _row("forces_full_compute_self_sec", lambda case: _fmt_time(_b(case, "forces_full_compute_self_sec"))),
        _row("target_local_force_total_sec", lambda case: _fmt_time(_b(case, "target_local_force_total_sec"))),
        _row("target_local_force_compute_self_sec", lambda case: _fmt_time(_b(case, "target_local_force_compute_self_sec"))),
        _row("device_sync_sec", lambda case: _fmt_time(_b(case, "device_sync_sec"))),
        _row("build_cell_list_sec", lambda case: _fmt_time(_b(case, "build_cell_list_sec"))),
        _row("zone_buffer_skin_sec", lambda case: _fmt_time(_b(case, "zone_buffer_skin_sec"))),
        _row("candidate_enum_sec", lambda case: _fmt_time(_b(case, "candidate_enum_sec"))),
        _row("zone_assign_sec", lambda case: _fmt_time(_b(case, "zone_assign_sec"))),
        _row("other_sec", lambda case: _fmt_time(_b(case, "other_sec"))),
        _row("forces_full_share", lambda case: _fmt_share(_b(case, "forces_full_share"))),
        _row("target_local_force_share", lambda case: _fmt_share(_b(case, "target_local_force_share"))),
        _row("device_sync_share", lambda case: _fmt_share(_b(case, "device_sync_share"))),
        _row("candidate_enum_share", lambda case: _fmt_share(_b(case, "candidate_enum_share"))),
        _row("zone_assign_share", lambda case: _fmt_share(_b(case, "zone_assign_share"))),
        _row(
            "wave_batch_enabled",
            lambda case: str(
                int(
                    bool(
                        dict(rows_by_case[case].get("breakdown", {}) or {})
                        .get("wave_batch_diagnostics", {})
                        .get("enabled", False)
                    )
                )
            ),
        ),
        _row(
            "wave_batch_launches_saved_per_step",
            lambda case: f"{float(dict(dict(rows_by_case[case].get('breakdown', {}) or {}).get('wave_batch_diagnostics', {})).get('launches_saved_per_step', 0.0) or 0.0):.3f}",
        ),
        _row(
            "wave_batch_avg_successful_wave_size",
            lambda case: f"{float(dict(dict(rows_by_case[case].get('breakdown', {}) or {}).get('wave_batch_diagnostics', {})).get('avg_successful_wave_size', 0.0) or 0.0):.3f}",
        ),
        _row(
            "wave_batch_neighbor_reuse_ratio",
            lambda case: _fmt_share(
                float(
                    dict(dict(rows_by_case[case].get("breakdown", {}) or {}).get("wave_batch_diagnostics", {})).get(
                        "neighbor_reuse_ratio_weighted", 0.0
                    )
                    or 0.0
                )
            ),
        ),
        _row(
            "wave_batch_candidate_union_to_target_ratio",
            lambda case: f"{float(dict(dict(rows_by_case[case].get('breakdown', {}) or {}).get('wave_batch_diagnostics', {})).get('candidate_union_to_target_ratio_avg', 0.0) or 0.0):.3f}",
        ),
        _row("forces_full_calls", lambda case: f"{int(round(_b(case, 'forces_full_calls')))}"),
        _row("forces_full_calls_per_step", lambda case: f"{_b(case, 'forces_full_calls_per_step'):.3f}"),
        _row("target_local_force_calls", lambda case: f"{int(round(_b(case, 'target_local_force_calls')))}"),
        _row("target_local_force_calls_per_step", lambda case: f"{_b(case, 'target_local_force_calls_per_step'):.3f}"),
        _row("device_sync_calls", lambda case: f"{int(round(_b(case, 'device_sync_calls')))}"),
        _row("device_sync_atoms_total", lambda case: f"{int(round(_b(case, 'device_sync_atoms_total')))}"),
        _row("avg_synced_atoms_per_call", lambda case: f"{_b(case, 'avg_synced_atoms_per_call'):.3f}"),
        _row("td_speedup_vs_space", lambda case: ("-" if case == "space_gpu" else _fmt_speedup(_speedup(rows_by_case["space_gpu"], rows_by_case["time_gpu"])))),
        "",
        "## Case Summary",
        "",
    ]
    for case in col_order:
        row = rows_by_case[case]
        breakdown = dict(row.get("breakdown", {}) or {})
        wave_diag = dict(breakdown.get("wave_batch_diagnostics", {}) or {})
        lines.append(
            f"- `{case}` ok={bool(row.get('ok', False))} effective=`{row.get('effective_device', '')}` "
            f"eval_scope=`{breakdown.get('many_body_evaluation_scope', '')}` "
            f"consume_scope=`{breakdown.get('many_body_consumption_scope', '')}` "
            f"median={_fmt_time(float(row.get('elapsed_sec_median', 0.0) or 0.0))} "
            f"forces_full_share={_fmt_share(float(breakdown.get('forces_full_share', 0.0) or 0.0))} "
            f"target_local_calls={int(round(float(breakdown.get('target_local_force_calls', 0.0) or 0.0)))} "
            f"wave_launches_saved_per_step={float(wave_diag.get('launches_saved_per_step', 0.0) or 0.0):.3f} "
            f"wave_reuse={_fmt_share(float(wave_diag.get('neighbor_reuse_ratio_weighted', 0.0) or 0.0))} "
            f"device_sync_share={_fmt_share(float(breakdown.get('device_sync_share', 0.0) or 0.0))} "
            f"error=`{row.get('error', '')}`"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Profile where EAM TD-vs-space GPU time is spent")
    ap.add_argument("--out", default="results/eam_td_breakdown_gpu.csv")
    ap.add_argument("--md", default="results/eam_td_breakdown_gpu.md")
    ap.add_argument("--json", default="results/eam_td_breakdown_gpu.summary.json")
    ap.add_argument("--n-atoms", type=int, default=10000)
    ap.add_argument("--steps", type=int, default=768)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
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

    rows = []
    for case, decomposition in (("space_gpu", "3d"), ("time_gpu", "1d")):
        rows.append(
            _run_case(
                label=case,
                decomposition=decomposition,
                requested_device="cuda",
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

    rows_by_case = {str(row["case"]): dict(row) for row in rows}
    report_markdown = _build_report(
        rows_by_case=rows_by_case,
        n_atoms=int(args.n_atoms),
        steps=int(args.steps),
        repeats=int(args.repeats),
        warmup=int(args.warmup),
        zones_total=int(args.zones_total),
        zones_nx=int(args.zones_nx),
        zones_ny=int(args.zones_ny),
        zones_nz=int(args.zones_nz),
    )
    td_speedup = _speedup(rows_by_case["space_gpu"], rows_by_case["time_gpu"])
    gpu_effective_ok = all(str(row.get("effective_device", "cpu")) == "cuda" for row in rows)
    ok_all = all(bool(row.get("ok", False)) for row in rows)
    if bool(args.require_effective_cuda):
        ok_all = bool(ok_all and gpu_effective_ok)

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
                "effective_device",
                "ok",
                "median_sec",
                "steps_per_sec",
                "many_body_evaluation_scope",
                "many_body_consumption_scope",
                "many_body_target_local_available",
                "forces_full_total_sec",
                "forces_full_compute_self_sec",
                "target_local_force_total_sec",
                "target_local_force_compute_self_sec",
                "device_sync_sec",
                "build_cell_list_sec",
                "zone_buffer_skin_sec",
                "candidate_enum_sec",
                "zone_assign_sec",
                "other_sec",
                "forces_full_calls",
                "forces_full_calls_per_step",
                "target_local_force_calls",
                "target_local_force_calls_per_step",
                "device_sync_calls",
                "device_sync_atoms_total",
                "avg_synced_atoms_per_call",
                "wave_batch_enabled",
                "wave_batch_launches_saved_per_step",
                "wave_batch_avg_successful_wave_size",
                "wave_batch_neighbor_reuse_ratio",
                "wave_batch_candidate_union_to_target_ratio",
            ]
        )
        for row in rows:
            breakdown = dict(row.get("breakdown", {}) or {})
            wave_diag = dict(breakdown.get("wave_batch_diagnostics", {}) or {})
            w.writerow(
                [
                    row["case"],
                    row["effective_device"],
                    int(bool(row["ok"])),
                    f"{float(row['elapsed_sec_median']):.6f}",
                    f"{float(row['steps_per_sec_median']):.6f}",
                    str(breakdown.get("many_body_evaluation_scope", "")),
                    str(breakdown.get("many_body_consumption_scope", "")),
                    int(bool(breakdown.get("many_body_target_local_available", 0))),
                    f"{float(breakdown.get('forces_full_total_sec', 0.0) or 0.0):.6f}",
                    f"{float(breakdown.get('forces_full_compute_self_sec', 0.0) or 0.0):.6f}",
                    f"{float(breakdown.get('target_local_force_total_sec', 0.0) or 0.0):.6f}",
                    f"{float(breakdown.get('target_local_force_compute_self_sec', 0.0) or 0.0):.6f}",
                    f"{float(breakdown.get('device_sync_sec', 0.0) or 0.0):.6f}",
                    f"{float(breakdown.get('build_cell_list_sec', 0.0) or 0.0):.6f}",
                    f"{float(breakdown.get('zone_buffer_skin_sec', 0.0) or 0.0):.6f}",
                    f"{float(breakdown.get('candidate_enum_sec', 0.0) or 0.0):.6f}",
                    f"{float(breakdown.get('zone_assign_sec', 0.0) or 0.0):.6f}",
                    f"{float(breakdown.get('other_sec', 0.0) or 0.0):.6f}",
                    f"{float(breakdown.get('forces_full_calls', 0.0) or 0.0):.3f}",
                    f"{float(breakdown.get('forces_full_calls_per_step', 0.0) or 0.0):.6f}",
                    f"{float(breakdown.get('target_local_force_calls', 0.0) or 0.0):.3f}",
                    f"{float(breakdown.get('target_local_force_calls_per_step', 0.0) or 0.0):.6f}",
                    f"{float(breakdown.get('device_sync_calls', 0.0) or 0.0):.3f}",
                    f"{float(breakdown.get('device_sync_atoms_total', 0.0) or 0.0):.3f}",
                    f"{float(breakdown.get('avg_synced_atoms_per_call', 0.0) or 0.0):.6f}",
                    int(bool(wave_diag.get("enabled", False))),
                    f"{float(wave_diag.get('launches_saved_per_step', 0.0) or 0.0):.6f}",
                    f"{float(wave_diag.get('avg_successful_wave_size', 0.0) or 0.0):.6f}",
                    f"{float(wave_diag.get('neighbor_reuse_ratio_weighted', 0.0) or 0.0):.6f}",
                    f"{float(wave_diag.get('candidate_union_to_target_ratio_avg', 0.0) or 0.0):.6f}",
                ]
            )

    out_md.write_text(report_markdown, encoding="utf-8")

    summary = {
        "n": len(rows),
        "total": len(rows),
        "ok": int(sum(1 for row in rows if bool(row.get("ok", False)))),
        "fail": int(sum(1 for row in rows if not bool(row.get("ok", False)))),
        "ok_all": bool(ok_all),
        "gpu_effective_ok": bool(gpu_effective_ok),
        "comparisons": {
            "td_speedup_gpu": td_speedup,
        },
        "force_scope_contract": {
            "version": "pr_mb03_v1",
            "baseline_reference_version": "pr_mb01_v1",
            "by_case": {
                str(row["case"]): {
                    "runtime_kind": str(dict(row.get("breakdown", {}) or {}).get("many_body_runtime_kind", "")),
                    "evaluation_scope": str(dict(row.get("breakdown", {}) or {}).get("many_body_evaluation_scope", "")),
                    "consumption_scope": str(dict(row.get("breakdown", {}) or {}).get("many_body_consumption_scope", "")),
                    "target_local_available": int(
                        bool(dict(row.get("breakdown", {}) or {}).get("many_body_target_local_available", 0))
                    ),
                }
                for row in rows
            },
        },
        "worst": {
            "max_elapsed_sec_median": max(
                (float(row.get("elapsed_sec_median", 0.0) or 0.0) for row in rows),
                default=0.0,
            ),
            "max_forces_full_share": max(
                (
                    float(dict(row.get("breakdown", {}) or {}).get("forces_full_share", 0.0) or 0.0)
                    for row in rows
                ),
                default=0.0,
            ),
            "max_device_sync_share": max(
                (
                    float(dict(row.get("breakdown", {}) or {}).get("device_sync_share", 0.0) or 0.0)
                    for row in rows
                ),
                default=0.0,
            ),
            "gpu_effective_ok": int(bool(gpu_effective_ok)),
        },
        "rows": rows,
        "by_case": rows_by_case,
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
    print(f"[eam-td-breakdown-gpu] wrote {out_csv}")
    print(f"[eam-td-breakdown-gpu] wrote {out_md}")
    print(f"[eam-td-breakdown-gpu] wrote {out_json}")
    if bool(args.strict) and not bool(summary["ok_all"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
