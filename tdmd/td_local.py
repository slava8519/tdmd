"""TD-local (single-process) reference implementation.

Execution paths:
  - sync_mode + many_body  → _run_sync_global()
  - sync_mode + 1d pair    → _run_sync_1d_zones()
  - async + 3d             → _run_async_3d()
  - async + 1d             → _run_async_1d()

All paths share a common ``_TDLocalCtx`` context object that bundles
runtime-resolved state (backend, ensemble, mass arrays, closures).
The public ``run_td_local()`` signature is preserved for backward compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np

from .atoms import normalize_atom_types, normalize_mass
from .backend import ComputeBackend, resolve_backend
from .celllist import build_cell_list
from .constants import FLOAT_EQ_ATOL, GEOM_EPSILON
from .ensembles import EnsembleSpec, apply_ensemble_step, build_ensemble_spec
from .force_dispatch import try_gpu_forces_on_targets
from .forces_cells import forces_on_targets_celllist_compact, forces_on_targets_zonecells
from .forces_gpu import (
    forces_on_targets_celllist_backend,
    forces_on_targets_pair_backend,
    mark_device_state_dirty,
    supports_pair_gpu,
)
from .integrator import vv_finish_velocities, vv_update_positions
from .many_body_scope import ManyBodyForceScope, td_local_many_body_force_scope
from .observer import emit_observer, observer_accepts_box
from .run_configs import TDLocalRunConfig
from .wavefront_1d import WAVEFRONT_1D_CONTRACT_VERSION, describe_wavefront_1d_zones
from .zone_bins_localz import PersistentZoneLocalZBinsCache
from .zones import (
    ZoneLayout1DCells,
    ZoneLayout3DBlocks,
    ZoneType,
    assign_atoms_to_zones,
    assign_atoms_to_zones_3d,
    compute_zone_buffer_skin,
    zones_overlapping_aabb_pbc,
    zones_overlapping_range_pbc,
)

TD_LOCAL_WAVE_BATCH_CONTRACT_VERSION = "pr_sw04_v1"
TD_LOCAL_WAVE_BATCH_DIAGNOSTICS_VERSION = "pr_sw05_v1"


@dataclass(frozen=True)
class TDLocalWaveBatchDiagnostics:
    version: str = TD_LOCAL_WAVE_BATCH_DIAGNOSTICS_VERSION
    runtime_contract_version: str = TD_LOCAL_WAVE_BATCH_CONTRACT_VERSION
    effective_device: str = "cpu"
    decomposition: str = "1d"
    sync_mode: bool = False
    eligible: bool = False
    enabled: bool = False
    many_body: bool = False
    pair_gpu: bool = False
    steps_total: int = 0
    candidate_multi_zone_waves_total: int = 0
    candidate_multi_zone_slots_total: int = 0
    attempted_wave_batches: int = 0
    successful_wave_batches: int = 0
    successful_batched_zones_total: int = 0
    cached_pre_force_hits: int = 0
    estimated_pre_force_launches_saved_total: int = 0
    avg_successful_wave_size: float = 0.0
    max_successful_wave_size: int = 0
    candidate_ids_naive_total: int = 0
    candidate_ids_union_total: int = 0
    candidate_ids_union_max: int = 0
    target_ids_union_total: int = 0
    target_ids_union_max: int = 0
    neighbor_reuse_atoms_total: int = 0
    neighbor_reuse_ratio_weighted: float = 0.0
    candidate_union_to_target_ratio_avg: float = 0.0
    candidate_union_to_target_ratio_max: float = 0.0
    launches_saved_per_step: float = 0.0
    attempted_wave_batches_per_step: float = 0.0
    successful_wave_batches_per_step: float = 0.0
    fallback_reason_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "version": str(self.version),
            "runtime_contract_version": str(self.runtime_contract_version),
            "effective_device": str(self.effective_device),
            "decomposition": str(self.decomposition),
            "sync_mode": bool(self.sync_mode),
            "eligible": bool(self.eligible),
            "enabled": bool(self.enabled),
            "many_body": bool(self.many_body),
            "pair_gpu": bool(self.pair_gpu),
            "steps_total": int(self.steps_total),
            "candidate_multi_zone_waves_total": int(self.candidate_multi_zone_waves_total),
            "candidate_multi_zone_slots_total": int(self.candidate_multi_zone_slots_total),
            "attempted_wave_batches": int(self.attempted_wave_batches),
            "successful_wave_batches": int(self.successful_wave_batches),
            "successful_batched_zones_total": int(self.successful_batched_zones_total),
            "cached_pre_force_hits": int(self.cached_pre_force_hits),
            "estimated_pre_force_launches_saved_total": int(
                self.estimated_pre_force_launches_saved_total
            ),
            "avg_successful_wave_size": float(self.avg_successful_wave_size),
            "max_successful_wave_size": int(self.max_successful_wave_size),
            "candidate_ids_naive_total": int(self.candidate_ids_naive_total),
            "candidate_ids_union_total": int(self.candidate_ids_union_total),
            "candidate_ids_union_max": int(self.candidate_ids_union_max),
            "target_ids_union_total": int(self.target_ids_union_total),
            "target_ids_union_max": int(self.target_ids_union_max),
            "neighbor_reuse_atoms_total": int(self.neighbor_reuse_atoms_total),
            "neighbor_reuse_ratio_weighted": float(self.neighbor_reuse_ratio_weighted),
            "candidate_union_to_target_ratio_avg": float(
                self.candidate_union_to_target_ratio_avg
            ),
            "candidate_union_to_target_ratio_max": float(
                self.candidate_union_to_target_ratio_max
            ),
            "launches_saved_per_step": float(self.launches_saved_per_step),
            "attempted_wave_batches_per_step": float(self.attempted_wave_batches_per_step),
            "successful_wave_batches_per_step": float(self.successful_wave_batches_per_step),
            "fallback_reason_counts": {
                str(key): int(value) for key, value in sorted(self.fallback_reason_counts.items())
            },
        }


@dataclass
class _WaveBatchDiagnosticsAccumulator:
    effective_device: str = "cpu"
    decomposition: str = "1d"
    sync_mode: bool = False
    eligible: bool = False
    enabled: bool = False
    many_body: bool = False
    pair_gpu: bool = False
    steps_total: int = 0
    candidate_multi_zone_waves_total: int = 0
    candidate_multi_zone_slots_total: int = 0
    attempted_wave_batches: int = 0
    successful_wave_batches: int = 0
    successful_batched_zones_total: int = 0
    cached_pre_force_hits: int = 0
    estimated_pre_force_launches_saved_total: int = 0
    successful_wave_size_total: int = 0
    max_successful_wave_size: int = 0
    candidate_ids_naive_total: int = 0
    candidate_ids_union_total: int = 0
    candidate_ids_union_max: int = 0
    target_ids_union_total: int = 0
    target_ids_union_max: int = 0
    neighbor_reuse_atoms_total: int = 0
    candidate_union_to_target_ratio_total: float = 0.0
    candidate_union_to_target_ratio_max: float = 0.0
    fallback_reason_counts: dict[str, int] = field(default_factory=dict)

    def note_fallback(self, reason: str) -> None:
        key = str(reason).strip()
        if not key:
            return
        self.fallback_reason_counts[key] = int(self.fallback_reason_counts.get(key, 0)) + 1

    def finalize(self) -> TDLocalWaveBatchDiagnostics:
        avg_wave_size = (
            float(self.successful_wave_size_total) / float(self.successful_wave_batches)
            if int(self.successful_wave_batches) > 0
            else 0.0
        )
        neighbor_reuse_ratio_weighted = (
            float(self.neighbor_reuse_atoms_total) / float(self.candidate_ids_naive_total)
            if int(self.candidate_ids_naive_total) > 0
            else 0.0
        )
        candidate_union_to_target_ratio_avg = (
            float(self.candidate_union_to_target_ratio_total) / float(self.successful_wave_batches)
            if int(self.successful_wave_batches) > 0
            else 0.0
        )
        launches_saved_per_step = (
            float(self.estimated_pre_force_launches_saved_total) / float(self.steps_total)
            if int(self.steps_total) > 0
            else 0.0
        )
        attempted_wave_batches_per_step = (
            float(self.attempted_wave_batches) / float(self.steps_total)
            if int(self.steps_total) > 0
            else 0.0
        )
        successful_wave_batches_per_step = (
            float(self.successful_wave_batches) / float(self.steps_total)
            if int(self.steps_total) > 0
            else 0.0
        )
        return TDLocalWaveBatchDiagnostics(
            effective_device=str(self.effective_device),
            decomposition=str(self.decomposition),
            sync_mode=bool(self.sync_mode),
            eligible=bool(self.eligible),
            enabled=bool(self.enabled),
            many_body=bool(self.many_body),
            pair_gpu=bool(self.pair_gpu),
            steps_total=int(self.steps_total),
            candidate_multi_zone_waves_total=int(self.candidate_multi_zone_waves_total),
            candidate_multi_zone_slots_total=int(self.candidate_multi_zone_slots_total),
            attempted_wave_batches=int(self.attempted_wave_batches),
            successful_wave_batches=int(self.successful_wave_batches),
            successful_batched_zones_total=int(self.successful_batched_zones_total),
            cached_pre_force_hits=int(self.cached_pre_force_hits),
            estimated_pre_force_launches_saved_total=int(
                self.estimated_pre_force_launches_saved_total
            ),
            avg_successful_wave_size=float(avg_wave_size),
            max_successful_wave_size=int(self.max_successful_wave_size),
            candidate_ids_naive_total=int(self.candidate_ids_naive_total),
            candidate_ids_union_total=int(self.candidate_ids_union_total),
            candidate_ids_union_max=int(self.candidate_ids_union_max),
            target_ids_union_total=int(self.target_ids_union_total),
            target_ids_union_max=int(self.target_ids_union_max),
            neighbor_reuse_atoms_total=int(self.neighbor_reuse_atoms_total),
            neighbor_reuse_ratio_weighted=float(neighbor_reuse_ratio_weighted),
            candidate_union_to_target_ratio_avg=float(candidate_union_to_target_ratio_avg),
            candidate_union_to_target_ratio_max=float(self.candidate_union_to_target_ratio_max),
            launches_saved_per_step=float(launches_saved_per_step),
            attempted_wave_batches_per_step=float(attempted_wave_batches_per_step),
            successful_wave_batches_per_step=float(successful_wave_batches_per_step),
            fallback_reason_counts={
                str(key): int(value) for key, value in sorted(self.fallback_reason_counts.items())
            },
        )


_LAST_TD_LOCAL_WAVE_BATCH_DIAGNOSTICS = TDLocalWaveBatchDiagnostics()


def _set_last_td_local_wave_batch_diagnostics(diag: TDLocalWaveBatchDiagnostics) -> None:
    global _LAST_TD_LOCAL_WAVE_BATCH_DIAGNOSTICS
    _LAST_TD_LOCAL_WAVE_BATCH_DIAGNOSTICS = diag


def get_last_td_local_wave_batch_diagnostics() -> TDLocalWaveBatchDiagnostics:
    return _LAST_TD_LOCAL_WAVE_BATCH_DIAGNOSTICS


def reset_td_local_wave_batch_diagnostics() -> None:
    _set_last_td_local_wave_batch_diagnostics(TDLocalWaveBatchDiagnostics())


def aggregate_td_local_wave_batch_diagnostics(
    runs: list[TDLocalWaveBatchDiagnostics | dict[str, object]],
) -> dict[str, object]:
    if not runs:
        return TDLocalWaveBatchDiagnostics().as_dict()
    normalized: list[dict[str, object]] = []
    for item in runs:
        if isinstance(item, TDLocalWaveBatchDiagnostics):
            normalized.append(item.as_dict())
        else:
            normalized.append(dict(item))
    last = normalized[-1]
    count_keys = [
        "steps_total",
        "candidate_multi_zone_waves_total",
        "candidate_multi_zone_slots_total",
        "attempted_wave_batches",
        "successful_wave_batches",
        "successful_batched_zones_total",
        "cached_pre_force_hits",
        "estimated_pre_force_launches_saved_total",
        "candidate_ids_naive_total",
        "candidate_ids_union_total",
        "target_ids_union_total",
        "neighbor_reuse_atoms_total",
    ]
    max_keys = [
        "max_successful_wave_size",
        "candidate_ids_union_max",
        "target_ids_union_max",
        "candidate_union_to_target_ratio_max",
    ]
    float_avg_keys = [
        "avg_successful_wave_size",
        "neighbor_reuse_ratio_weighted",
        "candidate_union_to_target_ratio_avg",
        "launches_saved_per_step",
        "attempted_wave_batches_per_step",
        "successful_wave_batches_per_step",
    ]
    out = {
        "version": str(last.get("version", TD_LOCAL_WAVE_BATCH_DIAGNOSTICS_VERSION)),
        "runtime_contract_version": str(
            last.get("runtime_contract_version", TD_LOCAL_WAVE_BATCH_CONTRACT_VERSION)
        ),
        "effective_device": str(last.get("effective_device", "cpu")),
        "decomposition": str(last.get("decomposition", "1d")),
        "sync_mode": bool(last.get("sync_mode", False)),
        "eligible": bool(last.get("eligible", False)),
        "enabled": bool(last.get("enabled", False)),
        "many_body": bool(last.get("many_body", False)),
        "pair_gpu": bool(last.get("pair_gpu", False)),
        "fallback_reason_counts": {},
    }
    for key in count_keys:
        values = [float(item.get(key, 0.0) or 0.0) for item in normalized]
        out[key] = int(round(sum(values) / float(len(values)))) if values else 0
    for key in max_keys:
        values = [float(item.get(key, 0.0) or 0.0) for item in normalized]
        max_value = max(values, default=0.0)
        out[key] = int(round(max_value)) if key != "candidate_union_to_target_ratio_max" else float(max_value)
    for key in float_avg_keys:
        values = [float(item.get(key, 0.0) or 0.0) for item in normalized]
        out[key] = (sum(values) / float(len(values))) if values else 0.0
    fallback_counts: dict[str, int] = {}
    for item in normalized:
        for key, value in dict(item.get("fallback_reason_counts", {}) or {}).items():
            fallback_counts[str(key)] = int(fallback_counts.get(str(key), 0)) + int(value)
    out["fallback_reason_counts"] = {
        str(key): int(value) for key, value in sorted(fallback_counts.items())
    }
    return out


# ---------------------------------------------------------------------------
# Internal context: resolved runtime state shared by all execution paths
# ---------------------------------------------------------------------------


@dataclass
class _TDLocalCtx:
    """Runtime context resolved once and shared by every execution path."""

    r: np.ndarray
    v: np.ndarray
    box: float
    potential: Any
    dt: float
    cutoff: float
    n_steps: int
    atom_types: np.ndarray
    mass: Union[float, np.ndarray]

    # resolved from mass
    mass_scalar: float | None
    mass_arr: np.ndarray | None
    inv_mass: float

    backend: ComputeBackend
    use_gpu_pair: bool
    ensemble: EnsembleSpec
    many_body: bool
    many_body_force_scope: ManyBodyForceScope | None
    rc_full: float
    wave_batch_diagnostics: _WaveBatchDiagnosticsAccumulator | None

    # observer
    observer: Any
    observer_every: int
    accepts_box: bool

    # trace
    trace: Any

    # chaos
    rng: np.random.Generator | None

    # zone / geometry
    cell_size: float
    zones_total: int
    zone_cells_w: int
    zone_cells_s: int
    zone_cells_pattern: Any
    traversal: str
    buffer_k: float
    skin_from_buffer: bool
    use_verlet: bool
    verlet_k_steps: int
    strict_min_zone_width: bool

    # 3D-specific
    zones_nx: int
    zones_ny: int
    zones_nz: int

    # ensemble raw params (for NPT scaling)
    ensemble_kind: str

    # ---- helpers --------------------------------------------------------

    def accel(self, f: np.ndarray, ids: Optional[np.ndarray] = None) -> np.ndarray:
        """Convert force → acceleration respecting per-atom mass."""
        if self.mass_arr is None:
            return f * self.inv_mass
        if ids is None:
            return f / self.mass_arr[:, None]
        return f / self.mass_arr[ids][:, None]

    def emit_observer(self, step: int) -> None:
        if not self.observer_every:
            return
        emit_observer(
            self.observer,
            accepts_box=self.accepts_box,
            step=step,
            r=self.r,
            v=self.v,
            box=float(self.box),
        )

    def apply_ensemble(self, step: int, *, atom_box: float) -> tuple[float, float]:
        new_box, _lam_t, lam_b = apply_ensemble_step(
            step=int(step),
            ensemble=self.ensemble,
            r=self.r,
            v=self.v,
            mass=(self.mass_scalar if self.mass_arr is None else self.mass_arr),
            box=float(atom_box),
            potential=self.potential,
            cutoff=self.cutoff,
            atom_types=self.atom_types,
            dt=self.dt,
        )
        return float(new_box), float(lam_b)

    def forces_full(self, rr: np.ndarray) -> np.ndarray:
        """Compute forces on *all* atoms using best available backend."""
        if self.backend.device == "cuda":
            ids_all = np.arange(rr.shape[0], dtype=np.int32)
            f_gpu = try_gpu_forces_on_targets(
                r=rr,
                box=self.box,
                cutoff=self.cutoff,
                rc=self.rc_full,
                potential=self.potential,
                target_ids=ids_all,
                candidate_ids=ids_all,
                atom_types=self.atom_types,
                backend=self.backend,
                prefer_marked_dirty=True,
            )
            if f_gpu is not None:
                return f_gpu
        if self.many_body:
            f_all, _pe, _w = self.potential.forces_energy_virial(
                rr, self.box, self.cutoff, self.atom_types
            )
            return np.asarray(f_all, dtype=float)
        ids_all = np.arange(rr.shape[0], dtype=np.int32)
        cell = build_cell_list(rr, ids_all, self.box, rc=self.rc_full)
        return forces_on_targets_celllist_compact(
            rr, self.box, self.potential, self.cutoff, ids_all, cell, atom_types=self.atom_types
        )


@dataclass
class _WaveBatchStepState:
    contract_version: str
    wavefront: dict[str, Any]
    wave_index_by_zid: dict[int, int]
    wave_zone_ids_by_zid: dict[int, tuple[int, ...]]
    attempted_wave_indices: set[int]
    pre_force_cache: dict[int, np.ndarray]
    pre_force_target_ids: dict[int, np.ndarray]


@dataclass(frozen=True)
class _WaveBatchComputeResult:
    forces_by_group: list[np.ndarray] | None
    union_targets_size: int = 0
    union_candidates_size: int = 0
    naive_candidate_size: int = 0
    fallback_reason: str = ""


# ---------------------------------------------------------------------------
# Zone scaling helpers (NPT)
# ---------------------------------------------------------------------------


def _scale_zones_1d(zones: list, lam: float, cutoff: float) -> None:
    if abs(float(lam) - 1.0) <= FLOAT_EQ_ATOL:
        return
    for z in zones:
        z.z0 = float(z.z0) * float(lam)
        z.z1 = float(z.z1) * float(lam)
    widths = [float(z.z1 - z.z0) for z in zones if getattr(z, "n_cells", 0) > 0 and z.z1 > z.z0]
    if widths and (min(widths) + GEOM_EPSILON < float(cutoff)):
        raise ValueError("NPT scaling violated zone width >= cutoff in td_local 1D layout")


def _scale_layout3(layout3: Any, lam: float, new_box: float, cutoff: float) -> None:
    if abs(float(lam) - 1.0) <= FLOAT_EQ_ATOL:
        layout3.box = float(new_box)
        return
    layout3.box = float(new_box)
    for z in layout3.zones:
        z.lo = np.asarray(z.lo, dtype=float) * float(lam)
        z.hi = np.asarray(z.hi, dtype=float) * float(lam)
    widths = [
        float(np.min(np.asarray(z.hi, dtype=float) - np.asarray(z.lo, dtype=float)))
        for z in layout3.zones
        if np.all(np.asarray(z.hi, dtype=float) > np.asarray(z.lo, dtype=float))
    ]
    if widths and (min(widths) + GEOM_EPSILON < float(cutoff)):
        raise ValueError("NPT scaling violated zone width >= cutoff in td_local 3D layout")


# ---------------------------------------------------------------------------
# Execution path 1: synchronous global (many_body or 3d sync)
# ---------------------------------------------------------------------------


def _run_sync_global(ctx: _TDLocalCtx) -> None:
    """Synchronous VV with global force evaluation (many-body or 3D sync)."""
    ids_all = np.arange(ctx.r.shape[0], dtype=np.int32)
    if ctx.observer is not None and ctx.observer_every:
        ctx.emit_observer(0)
    for step in range(1, ctx.n_steps + 1):
        f0 = ctx.forces_full(ctx.r)
        v_half = ctx.v + 0.5 * ctx.dt * ctx.accel(f0, ids_all)
        ctx.r[:] = (ctx.r + ctx.dt * v_half) % ctx.box
        if ctx.backend.device == "cuda":
            mark_device_state_dirty(ctx.r, ids_all)
        f1 = ctx.forces_full(ctx.r)
        ctx.v[:] = v_half + 0.5 * ctx.dt * ctx.accel(f1, ids_all)
        ctx.box, _lam_b = ctx.apply_ensemble(step, atom_box=ctx.box)
        if ctx.backend.device == "cuda":
            mark_device_state_dirty(ctx.r, ids_all)
        if ctx.observer is not None and ctx.observer_every and (step % ctx.observer_every == 0):
            ctx.emit_observer(step)


# ---------------------------------------------------------------------------
# Execution path 2: synchronous 1D zone-by-zone
# ---------------------------------------------------------------------------


def _run_sync_1d_zones(ctx: _TDLocalCtx) -> None:
    """Synchronous VV with zone-level force evaluation (1D pair potentials)."""
    layout = ZoneLayout1DCells(
        box=ctx.box,
        cell_size=ctx.cell_size,
        zones_total=ctx.zones_total,
        pattern_cells=ctx.zone_cells_pattern,
        zone_cells_w=ctx.zone_cells_w,
        zone_cells_s=ctx.zone_cells_s,
        min_zone_width=float(ctx.cutoff),
        strict_min_width=bool(ctx.strict_min_zone_width),
    )
    zones = layout.build()
    if ctx.traversal == "backward":
        order = list(range(ctx.zones_total - 1, -1, -1))
    else:
        order = list(range(ctx.zones_total))
    cache = PersistentZoneLocalZBinsCache()

    if ctx.observer is not None and ctx.observer_every:
        ctx.emit_observer(0)

    for step in range(1, ctx.n_steps + 1):
        assign_atoms_to_zones(ctx.r, zones, ctx.box)
        for z in zones:
            z.ztype = ZoneType.D if z.atom_ids.size else ZoneType.F

        rc = float(ctx.cutoff)
        f0 = np.zeros_like(ctx.r)
        for zid in order:
            z = zones[zid]
            if z.atom_ids.size == 0:
                continue
            z0p = z.z0 - ctx.cutoff
            z1p = z.z1 + ctx.cutoff
            pzids = zones_overlapping_range_pbc(z0p, z1p, ctx.box, zones)
            cand = []
            for nzid in pzids:
                nz = zones[nzid]
                if nz.atom_ids.size:
                    cand.append(nz.atom_ids)
            candidate_ids = np.concatenate(cand) if cand else np.empty((0,), np.int32)
            if candidate_ids.size:
                f_zone = _forces_on_zone_1d(
                    ctx, z, candidate_ids, cache, rc, 0.0, 2 * step, 1, z0p, z1p
                )
                f0[z.atom_ids] = f_zone

        v_half = ctx.v + 0.5 * ctx.dt * ctx.accel(f0)
        ctx.r[:] = (ctx.r + ctx.dt * v_half) % ctx.box
        if ctx.backend.device == "cuda":
            mark_device_state_dirty(ctx.r)

        assign_atoms_to_zones(ctx.r, zones, ctx.box)
        for z in zones:
            z.ztype = ZoneType.D if z.atom_ids.size else ZoneType.F

        f1 = np.zeros_like(ctx.r)
        for zid in order:
            z = zones[zid]
            if z.atom_ids.size == 0:
                continue
            z0p = z.z0 - ctx.cutoff
            z1p = z.z1 + ctx.cutoff
            pzids = zones_overlapping_range_pbc(z0p, z1p, ctx.box, zones)
            cand = []
            for nzid in pzids:
                nz = zones[nzid]
                if nz.atom_ids.size:
                    cand.append(nz.atom_ids)
            candidate_ids = np.concatenate(cand) if cand else np.empty((0,), np.int32)
            if candidate_ids.size:
                f_zone = _forces_on_zone_1d(
                    ctx, z, candidate_ids, cache, rc, 0.0, 2 * step + 1, 1, z0p, z1p
                )
                f1[z.atom_ids] = f_zone

        ctx.v[:] = v_half + 0.5 * ctx.dt * ctx.accel(f1)
        ctx.box, lam_b = ctx.apply_ensemble(step, atom_box=ctx.box)
        if ctx.backend.device == "cuda":
            mark_device_state_dirty(ctx.r)
        if ctx.ensemble_kind == "npt":
            _scale_zones_1d(zones, lam_b, ctx.cutoff)
        if ctx.observer is not None and ctx.observer_every and (step % ctx.observer_every == 0):
            ctx.emit_observer(step)


# ---------------------------------------------------------------------------
# Execution path 3: async 3D block zones
# ---------------------------------------------------------------------------


def _run_async_3d(ctx: _TDLocalCtx) -> None:
    """Asynchronous zone-by-zone VV with 3D block decomposition."""
    layout3 = ZoneLayout3DBlocks.build(
        box=ctx.box, nx=int(ctx.zones_nx), ny=int(ctx.zones_ny), nz=int(ctx.zones_nz)
    )
    zones3 = layout3.zones
    assign_atoms_to_zones_3d(ctx.r, layout3)
    for z in zones3:
        if z.atom_ids.size:
            z.ztype = ZoneType.D

    order = list(range(len(zones3)))
    if ctx.traversal == "backward":
        order = list(range(len(zones3) - 1, -1, -1))

    if ctx.observer is not None and ctx.observer_every:
        ctx.emit_observer(0)

    for step in range(1, ctx.n_steps + 1):
        if ctx.rng is not None:
            ctx.rng.shuffle(order)

        for z in zones3:
            z.ztype = ZoneType.D if z.atom_ids.size else ZoneType.F
        processed = np.zeros(ctx.r.shape[0], dtype=bool)

        skin_global = 0.0
        for z in zones3:
            b, skin = compute_zone_buffer_skin(
                ctx.v, z.atom_ids, ctx.dt, ctx.buffer_k, skin_from_buffer=ctx.skin_from_buffer
            )
            z.buffer = b
            z.skin = skin
            skin_global = max(skin_global, skin)
        rc = ctx.cutoff + skin_global

        ids_all = np.arange(ctx.r.shape[0], dtype=np.int32)
        cell = build_cell_list(ctx.r, ids_all, ctx.box, rc=max(rc, GEOM_EPSILON))

        for zid in order:
            z = zones3[zid]
            if z.ztype in (ZoneType.D, ZoneType.P) and z.atom_ids.size:
                ids0 = z.atom_ids[~processed[z.atom_ids]]
                if ids0.size == 0:
                    z.ztype = ZoneType.S
                    continue
                state_before = z.ztype
                z.ztype = ZoneType.W
                if ctx.trace is not None:
                    ctx.trace.log(
                        step_id=int(step),
                        zone_id=int(zid),
                        event="START_COMPUTE",
                        state_before=state_before,
                        state_after=z.ztype,
                        halo_ids_count=0,
                        migration_count=0,
                        lag=0,
                        invariant_flags="",
                    )

                lo = z.lo - ctx.cutoff
                hi = z.hi + ctx.cutoff
                deps = zones_overlapping_aabb_pbc(lo, hi, ctx.box, layout3)
                cand = []
                for did in deps:
                    dz = zones3[did]
                    if dz.atom_ids.size:
                        if dz.ztype == ZoneType.D and did != zid:
                            dz.ztype = ZoneType.P
                        cand.append(dz.atom_ids)
                candidate_ids = np.concatenate(cand) if cand else np.empty((0,), np.int32)

                if candidate_ids.size or ctx.many_body:
                    f = _forces_3d(ctx, ids0, ids_all, cell, rc, candidate_ids)
                    vv_update_positions(ctx.r, ctx.v, ctx.mass, ctx.dt, ctx.box, ids0, f)
                    if ctx.backend.device == "cuda":
                        mark_device_state_dirty(ctx.r, ids0)
                    processed[ids0] = True
                    assign_atoms_to_zones_3d(ctx.r, layout3)

                    cand2 = []
                    for did in deps:
                        dz = zones3[did]
                        if dz.atom_ids.size:
                            cand2.append(dz.atom_ids)
                    candidate_ids2 = np.concatenate(cand2) if cand2 else np.empty((0,), np.int32)

                    f2 = _forces_3d_post(ctx, ids0, ids_all, cell, rc, candidate_ids2)
                    vv_finish_velocities(ctx.v, ctx.mass, ctx.dt, ids0, f2)

                state_before = z.ztype
                z.ztype = ZoneType.S
                if ctx.trace is not None:
                    ctx.trace.log(
                        step_id=int(step),
                        zone_id=int(zid),
                        event="FINISH_COMPUTE",
                        state_before=state_before,
                        state_after=z.ztype,
                        halo_ids_count=0,
                        migration_count=0,
                        lag=0,
                        invariant_flags="",
                    )

        ctx.box, lam_b = ctx.apply_ensemble(step, atom_box=ctx.box)
        if ctx.backend.device == "cuda":
            mark_device_state_dirty(ctx.r)
        if ctx.ensemble_kind == "npt":
            _scale_layout3(layout3, lam_b, ctx.box, ctx.cutoff)

        if ctx.observer is not None and ctx.observer_every and (step % ctx.observer_every == 0):
            ctx.emit_observer(step)


# ---------------------------------------------------------------------------
# Execution path 4: async 1D slab zones (legacy, fully optimized)
# ---------------------------------------------------------------------------


def _run_async_1d(ctx: _TDLocalCtx) -> None:
    """Asynchronous zone-by-zone VV with 1D slab decomposition (optimized)."""
    layout = ZoneLayout1DCells(
        box=ctx.box,
        cell_size=ctx.cell_size,
        zones_total=ctx.zones_total,
        pattern_cells=ctx.zone_cells_pattern,
        zone_cells_w=ctx.zone_cells_w,
        zone_cells_s=ctx.zone_cells_s,
        min_zone_width=float(ctx.cutoff),
        strict_min_width=bool(ctx.strict_min_zone_width),
    )
    zones = layout.build()
    assign_atoms_to_zones(ctx.r, zones, ctx.box)
    for z in zones:
        z.ztype = ZoneType.D

    if ctx.traversal == "backward":
        order = list(range(ctx.zones_total - 1, -1, -1))
    else:
        order = list(range(ctx.zones_total))

    cache = PersistentZoneLocalZBinsCache()

    if ctx.observer is not None and ctx.observer_every:
        ctx.emit_observer(0)

    for step in range(1, ctx.n_steps + 1):
        if ctx.rng is not None:
            ctx.rng.shuffle(order)
        for z in zones:
            z.ztype = ZoneType.D if z.atom_ids.size else ZoneType.F
        processed = np.zeros(ctx.r.shape[0], dtype=bool)
        skin_global = 0.0
        for z in zones:
            b, skin = compute_zone_buffer_skin(
                ctx.v, z.atom_ids, ctx.dt, ctx.buffer_k, skin_from_buffer=ctx.skin_from_buffer
            )
            z.buffer = b
            z.skin = skin
            skin_global = max(skin_global, skin)
        rc = ctx.cutoff + skin_global
        wave_batch = _build_async_1d_wave_batch_state(ctx=ctx, zones=zones, order=order)

        for zid in order:
            z = zones[zid]
            if z.ztype in (ZoneType.D, ZoneType.P) and z.atom_ids.size:
                ids0 = z.atom_ids[~processed[z.atom_ids]]
                if ids0.size == 0:
                    z.ztype = ZoneType.S
                    continue
                state_before = z.ztype
                z.ztype = ZoneType.W
                if ctx.trace is not None:
                    ctx.trace.log(
                        step_id=int(step),
                        zone_id=int(zid),
                        event="START_COMPUTE",
                        state_before=state_before,
                        state_after=z.ztype,
                        halo_ids_count=0,
                        migration_count=0,
                        lag=0,
                        invariant_flags="",
                    )
                z0p = z.z0 - ctx.cutoff
                z1p = z.z1 + ctx.cutoff
                pzids = zones_overlapping_range_pbc(z0p, z1p, ctx.box, zones)
                cand = []
                for nzid in pzids:
                    nz = zones[nzid]
                    if nz.atom_ids.size:
                        if nz.ztype == ZoneType.D and nzid != zid:
                            nz.ztype = ZoneType.P
                        cand.append(nz.atom_ids)
                candidate_ids = np.concatenate(cand) if cand else np.empty((0,), np.int32)

                if candidate_ids.size or ctx.many_body:
                    # ---- position update (VV half-step 1) ----
                    cached_pre = _pop_wave_batch_pre_force(
                        wave_batch=wave_batch,
                        ctx=ctx,
                        zones=zones,
                        zid=int(zid),
                        processed=processed,
                        rc=float(rc),
                    )
                    if cached_pre is not None:
                        f = cached_pre
                    elif ctx.many_body:
                        f = _forces_many_body_targets(ctx, ids0, candidate_ids, rc)
                    else:
                        f = _forces_on_zone_1d_async(
                            ctx, ids0, candidate_ids, cache, zid, rc, skin_global, step, z0p, z1p
                        )
                    vv_update_positions(ctx.r, ctx.v, ctx.mass, ctx.dt, ctx.box, ids0, f)
                    if ctx.backend.device == "cuda":
                        mark_device_state_dirty(ctx.r, ids0)
                    assign_atoms_to_zones(ctx.r, zones, ctx.box)

                    # ---- velocity finish (VV half-step 2) ----
                    cand2 = []
                    for nzid in pzids:
                        nz = zones[nzid]
                        if nz.atom_ids.size:
                            cand2.append(nz.atom_ids)
                    candidate_ids2 = np.concatenate(cand2) if cand2 else np.empty((0,), np.int32)
                    if ctx.many_body:
                        f2 = _forces_many_body_targets(ctx, ids0, candidate_ids2, rc)
                    elif candidate_ids2.size:
                        f2 = _forces_on_zone_1d_async(
                            ctx, ids0, candidate_ids2, cache, zid, rc, skin_global, step, z0p, z1p
                        )
                    else:
                        f2 = np.zeros((ids0.size, 3), dtype=np.float64)
                    vv_finish_velocities(ctx.v, ctx.mass, ctx.dt, ids0, f2)

                state_before = z.ztype
                z.ztype = ZoneType.S
                if ctx.trace is not None:
                    ctx.trace.log(
                        step_id=int(step),
                        zone_id=int(zid),
                        event="FINISH_COMPUTE",
                        state_before=state_before,
                        state_after=z.ztype,
                        halo_ids_count=0,
                        migration_count=0,
                        lag=0,
                        invariant_flags="",
                    )

        ctx.box, lam_b = ctx.apply_ensemble(step, atom_box=ctx.box)
        if ctx.backend.device == "cuda":
            mark_device_state_dirty(ctx.r)
        if ctx.ensemble_kind == "npt":
            _scale_zones_1d(zones, lam_b, ctx.cutoff)

        if ctx.observer is not None and ctx.observer_every and (step % ctx.observer_every == 0):
            ctx.emit_observer(step)


# ---------------------------------------------------------------------------
# Async 1D wave-batch helpers (GPU refinement only)
# ---------------------------------------------------------------------------


def _wave_batch_runtime_enabled(ctx: _TDLocalCtx) -> bool:
    return bool(ctx.backend.device == "cuda" and (ctx.use_gpu_pair or ctx.many_body))


def _build_async_1d_wave_batch_state(
    *,
    ctx: _TDLocalCtx,
    zones: list[Any],
    order: list[int],
) -> _WaveBatchStepState | None:
    if not _wave_batch_runtime_enabled(ctx):
        return None
    wavefront = describe_wavefront_1d_zones(
        zones=zones,
        box=float(ctx.box),
        cutoff=float(ctx.cutoff),
        traversal_order=list(order),
    )
    candidate_waves = list(wavefront.get("candidate_waves", []) or [])
    diag = ctx.wave_batch_diagnostics
    multi_zone_waves = [
        tuple(int(v) for v in list(dict(wave).get("zone_ids", [])))
        for wave in candidate_waves
        if len(list(dict(wave).get("zone_ids", []))) > 1
    ]
    if diag is not None:
        diag.enabled = bool(multi_zone_waves)
        diag.candidate_multi_zone_waves_total += int(len(multi_zone_waves))
        diag.candidate_multi_zone_slots_total += int(sum(len(zone_ids) for zone_ids in multi_zone_waves))
    if not candidate_waves:
        return None
    wave_index_by_zid: dict[int, int] = {}
    wave_zone_ids_by_zid: dict[int, tuple[int, ...]] = {}
    for wave in candidate_waves:
        wave_index = int(dict(wave).get("wave_index", 0) or 0)
        zone_ids = tuple(int(v) for v in list(dict(wave).get("zone_ids", [])))
        if len(zone_ids) <= 1:
            continue
        for zid in zone_ids:
            wave_index_by_zid[int(zid)] = int(wave_index)
            wave_zone_ids_by_zid[int(zid)] = tuple(zone_ids)
    if not wave_zone_ids_by_zid:
        return None
    return _WaveBatchStepState(
        contract_version=TD_LOCAL_WAVE_BATCH_CONTRACT_VERSION,
        wavefront=dict(wavefront),
        wave_index_by_zid=wave_index_by_zid,
        wave_zone_ids_by_zid=wave_zone_ids_by_zid,
        attempted_wave_indices=set(),
        pre_force_cache={},
        pre_force_target_ids={},
    )


def _candidate_ids_for_zone_1d(
    *,
    ctx: _TDLocalCtx,
    zones: list[Any],
    zid: int,
) -> tuple[np.ndarray, list[int]]:
    z = zones[int(zid)]
    z0p = float(z.z0) - float(ctx.cutoff)
    z1p = float(z.z1) + float(ctx.cutoff)
    pzids = zones_overlapping_range_pbc(z0p, z1p, ctx.box, zones)
    cand: list[np.ndarray] = []
    for nzid in pzids:
        nz = zones[int(nzid)]
        if nz.atom_ids.size:
            cand.append(np.asarray(nz.atom_ids, dtype=np.int32))
    if cand:
        return np.concatenate(cand).astype(np.int32, copy=False), [int(v) for v in pzids]
    return np.empty((0,), dtype=np.int32), [int(v) for v in pzids]


def _compute_wave_batched_pre_forces(
    *,
    ctx: _TDLocalCtx,
    target_groups: list[np.ndarray],
    candidate_groups: list[np.ndarray],
    rc: float,
) -> _WaveBatchComputeResult:
    if not target_groups:
        return _WaveBatchComputeResult([])
    union_targets = np.unique(np.concatenate(target_groups).astype(np.int32))
    naive_candidate_size = int(sum(int(np.asarray(group, dtype=np.int32).size) for group in candidate_groups))
    if int(union_targets.size) != sum(int(group.size) for group in target_groups):
        return _WaveBatchComputeResult(
            None,
            union_targets_size=int(union_targets.size),
            fallback_reason="target_overlap",
        )
    candidate_arrays = [group for group in candidate_groups if int(group.size) > 0]
    union_candidates = (
        np.unique(np.concatenate(candidate_arrays + [union_targets]).astype(np.int32))
        if candidate_arrays
        else np.asarray(union_targets, dtype=np.int32)
    )
    if ctx.many_body:
        f_union = try_gpu_forces_on_targets(
            r=ctx.r,
            box=float(ctx.box),
            cutoff=float(ctx.cutoff),
            rc=float(rc),
            potential=ctx.potential,
            target_ids=union_targets,
            candidate_ids=union_candidates,
            atom_types=ctx.atom_types,
            backend=ctx.backend,
            prefer_marked_dirty=True,
        )
    else:
        f_union = forces_on_targets_pair_backend(
            r=ctx.r,
            box=float(ctx.box),
            cutoff=float(ctx.cutoff),
            potential=ctx.potential,
            target_ids=union_targets,
            candidate_ids=union_candidates,
            atom_types=ctx.atom_types,
            backend=ctx.backend,
            prefer_marked_dirty=True,
        )
    if f_union is None:
        return _WaveBatchComputeResult(
            None,
            union_targets_size=int(union_targets.size),
            union_candidates_size=int(union_candidates.size),
            naive_candidate_size=int(naive_candidate_size),
            fallback_reason="gpu_backend_returned_none",
        )
    out_union = np.asarray(f_union, dtype=np.float64)
    union_pos = {int(gid): int(i) for i, gid in enumerate(union_targets.tolist())}
    out: list[np.ndarray] = []
    for group in target_groups:
        idx = [union_pos[int(gid)] for gid in np.asarray(group, dtype=np.int32).tolist()]
        out.append(np.asarray(out_union[idx], dtype=np.float64))
    return _WaveBatchComputeResult(
        out,
        union_targets_size=int(union_targets.size),
        union_candidates_size=int(union_candidates.size),
        naive_candidate_size=int(naive_candidate_size),
    )


def _prepare_wave_batch_pre_forces(
    *,
    wave_batch: _WaveBatchStepState,
    ctx: _TDLocalCtx,
    zones: list[Any],
    processed: np.ndarray,
    zid: int,
    rc: float,
) -> None:
    diag = ctx.wave_batch_diagnostics
    wave_index = wave_batch.wave_index_by_zid.get(int(zid))
    if wave_index is None or int(wave_index) in wave_batch.attempted_wave_indices:
        return
    wave_batch.attempted_wave_indices.add(int(wave_index))
    zone_ids = list(wave_batch.wave_zone_ids_by_zid.get(int(zid), ()))
    if len(zone_ids) <= 1:
        return
    if diag is not None:
        diag.attempted_wave_batches += 1

    batched_zone_ids: list[int] = []
    target_groups: list[np.ndarray] = []
    candidate_groups: list[np.ndarray] = []
    for wzid in zone_ids:
        zone = zones[int(wzid)]
        allowed_state = zone.ztype in (ZoneType.D, ZoneType.P) or (
            int(wzid) == int(zid) and zone.ztype == ZoneType.W
        )
        if not allowed_state or not zone.atom_ids.size:
            continue
        target_ids = np.asarray(zone.atom_ids[~processed[zone.atom_ids]], dtype=np.int32)
        if target_ids.size == 0:
            continue
        candidate_ids, _pzids = _candidate_ids_for_zone_1d(ctx=ctx, zones=zones, zid=int(wzid))
        if target_ids.size == 0 or (candidate_ids.size == 0 and not ctx.many_body):
            continue
        batched_zone_ids.append(int(wzid))
        target_groups.append(np.asarray(target_ids, dtype=np.int32))
        candidate_groups.append(np.asarray(candidate_ids, dtype=np.int32))

    if len(batched_zone_ids) <= 1:
        if diag is not None:
            diag.note_fallback("insufficient_ready_zones")
        return
    result = _compute_wave_batched_pre_forces(
        ctx=ctx,
        target_groups=target_groups,
        candidate_groups=candidate_groups,
        rc=float(rc),
    )
    if result.forces_by_group is None:
        if diag is not None:
            diag.note_fallback(str(result.fallback_reason))
        return
    if diag is not None:
        wave_size = int(len(batched_zone_ids))
        diag.successful_wave_batches += 1
        diag.successful_batched_zones_total += int(wave_size)
        diag.successful_wave_size_total += int(wave_size)
        diag.max_successful_wave_size = max(int(diag.max_successful_wave_size), int(wave_size))
        diag.estimated_pre_force_launches_saved_total += max(0, int(wave_size) - 1)
        diag.candidate_ids_naive_total += int(result.naive_candidate_size)
        diag.candidate_ids_union_total += int(result.union_candidates_size)
        diag.candidate_ids_union_max = max(
            int(diag.candidate_ids_union_max), int(result.union_candidates_size)
        )
        diag.target_ids_union_total += int(result.union_targets_size)
        diag.target_ids_union_max = max(int(diag.target_ids_union_max), int(result.union_targets_size))
        diag.neighbor_reuse_atoms_total += max(
            0, int(result.naive_candidate_size) - int(result.union_candidates_size)
        )
        if int(result.union_targets_size) > 0:
            union_to_target = float(result.union_candidates_size) / float(result.union_targets_size)
            diag.candidate_union_to_target_ratio_total += float(union_to_target)
            diag.candidate_union_to_target_ratio_max = max(
                float(diag.candidate_union_to_target_ratio_max),
                float(union_to_target),
            )
    for wzid, target_ids, forces in zip(batched_zone_ids, target_groups, result.forces_by_group):
        wave_batch.pre_force_target_ids[int(wzid)] = np.asarray(target_ids, dtype=np.int32).copy()
        wave_batch.pre_force_cache[int(wzid)] = np.asarray(forces, dtype=np.float64).copy()


def _pop_wave_batch_pre_force(
    *,
    wave_batch: _WaveBatchStepState | None,
    ctx: _TDLocalCtx,
    zones: list[Any],
    zid: int,
    processed: np.ndarray,
    rc: float,
) -> np.ndarray | None:
    if wave_batch is None:
        return None
    if int(zid) not in wave_batch.pre_force_cache:
        _prepare_wave_batch_pre_forces(
            wave_batch=wave_batch,
            ctx=ctx,
            zones=zones,
            processed=processed,
            zid=int(zid),
            rc=float(rc),
        )
    cached_target_ids = wave_batch.pre_force_target_ids.get(int(zid))
    if cached_target_ids is None:
        return None
    current_target_ids = np.asarray(
        zones[int(zid)].atom_ids[~processed[zones[int(zid)].atom_ids]], dtype=np.int32
    )
    if not np.array_equal(np.asarray(cached_target_ids, dtype=np.int32), current_target_ids):
        if ctx.wave_batch_diagnostics is not None:
            ctx.wave_batch_diagnostics.note_fallback("cached_target_ids_changed")
        wave_batch.pre_force_target_ids.pop(int(zid), None)
        wave_batch.pre_force_cache.pop(int(zid), None)
        return None
    wave_batch.pre_force_target_ids.pop(int(zid), None)
    if ctx.wave_batch_diagnostics is not None:
        ctx.wave_batch_diagnostics.cached_pre_force_hits += 1
    return np.asarray(wave_batch.pre_force_cache.pop(int(zid)), dtype=np.float64)


# ---------------------------------------------------------------------------
# Force helpers (shared by execution paths)
# ---------------------------------------------------------------------------


def _forces_on_zone_1d(
    ctx: _TDLocalCtx,
    z: Any,
    candidate_ids: np.ndarray,
    cache: PersistentZoneLocalZBinsCache,
    rc: float,
    skin_global: float,
    step: int,
    verlet_k: int,
    z0p: float,
    z1p: float,
) -> np.ndarray:
    """Compute forces on a zone for sync 1D path (GPU or CPU cell-list)."""
    if ctx.use_gpu_pair:
        f_zone = forces_on_targets_pair_backend(
            r=ctx.r,
            box=ctx.box,
            cutoff=ctx.cutoff,
            potential=ctx.potential,
            target_ids=z.atom_ids,
            candidate_ids=candidate_ids,
            atom_types=ctx.atom_types,
            backend=ctx.backend,
            prefer_marked_dirty=True,
        )
        if f_zone is None:
            f_zone = np.zeros((z.atom_ids.size, 3), dtype=np.float64)
        return f_zone
    zc = cache.get(
        z.zid if hasattr(z, "zid") else 0,
        ctx.r,
        ctx.box,
        candidate_ids,
        rc=rc,
        skin_global=skin_global,
        step=step,
        verlet_k_steps=verlet_k,
        z0=z0p,
        z1=z1p,
    )
    return forces_on_targets_zonecells(
        ctx.r, ctx.box, ctx.potential, ctx.cutoff, z.atom_ids, zc, atom_types=ctx.atom_types
    )


def _forces_on_zone_1d_async(
    ctx: _TDLocalCtx,
    ids0: np.ndarray,
    candidate_ids: np.ndarray,
    cache: PersistentZoneLocalZBinsCache,
    zid: int,
    rc: float,
    skin_global: float,
    step: int,
    z0p: float,
    z1p: float,
) -> np.ndarray:
    """Compute forces for async 1D path (GPU pair or CPU zone-cells)."""
    if ctx.use_gpu_pair:
        f = forces_on_targets_pair_backend(
            r=ctx.r,
            box=ctx.box,
            cutoff=ctx.cutoff,
            potential=ctx.potential,
            target_ids=ids0,
            candidate_ids=candidate_ids,
            atom_types=ctx.atom_types,
            backend=ctx.backend,
            prefer_marked_dirty=True,
        )
        if f is not None:
            return f
    # CPU zone-cell fallback
    zc = cache.get(
        zid,
        ctx.r,
        ctx.box,
        candidate_ids,
        rc=rc,
        skin_global=(skin_global if ctx.use_verlet else 0.0),
        step=step,
        verlet_k_steps=(ctx.verlet_k_steps if ctx.use_verlet else 1),
        z0=z0p,
        z1=z1p,
    )
    return forces_on_targets_zonecells(
        ctx.r, ctx.box, ctx.potential, ctx.cutoff, ids0, zc, atom_types=ctx.atom_types
    )


def _forces_many_body_targets(
    ctx: _TDLocalCtx,
    target_ids: np.ndarray,
    candidate_ids: np.ndarray,
    rc: float,
) -> np.ndarray:
    """Many-body target-force evaluation for async td_local paths.

    Contract:
    - CPU path is the reference semantics and uses target-local
      `potential.forces_on_targets(...)` when available.
    - CUDA path first tries the target/candidate-local GPU refinement path.
    - If the CUDA target-local refinement is unavailable for a call, it falls back to
      the existing full-system GPU evaluation instead of silently falling back to CPU.
    """
    if ctx.backend.device == "cuda":
        f_gpu = try_gpu_forces_on_targets(
            r=ctx.r,
            box=ctx.box,
            cutoff=ctx.cutoff,
            rc=float(rc),
            potential=ctx.potential,
            target_ids=np.asarray(target_ids, dtype=np.int32),
            candidate_ids=np.asarray(candidate_ids, dtype=np.int32),
            atom_types=ctx.atom_types,
            backend=ctx.backend,
            prefer_marked_dirty=True,
        )
        if f_gpu is not None:
            return np.asarray(f_gpu, dtype=float)
        return ctx.forces_full(ctx.r)[target_ids]
    if hasattr(ctx.potential, "forces_on_targets"):
        try:
            f = ctx.potential.forces_on_targets(
                r=ctx.r,
                box=ctx.box,
                cutoff=ctx.cutoff,
                rc=float(rc),
                atom_types=ctx.atom_types,
                target_ids=target_ids,
                candidate_ids=candidate_ids,
            )
        except TypeError:
            f = ctx.potential.forces_on_targets(
                r=ctx.r,
                box=ctx.box,
                cutoff=ctx.cutoff,
                atom_types=ctx.atom_types,
                target_ids=target_ids,
                candidate_ids=candidate_ids,
            )
        return np.asarray(f, dtype=float)
    return ctx.forces_full(ctx.r)[target_ids]


def _forces_3d(
    ctx: _TDLocalCtx,
    ids0: np.ndarray,
    ids_all: np.ndarray,
    cell: Any,
    rc: float,
    candidate_ids: np.ndarray,
) -> np.ndarray:
    """Force computation for async 3D path (position-update half)."""
    if ctx.many_body:
        return _forces_many_body_targets(ctx, ids0, candidate_ids, rc)
    if ctx.use_gpu_pair:
        f = forces_on_targets_celllist_backend(
            r=ctx.r,
            box=ctx.box,
            cutoff=ctx.cutoff,
            rc=max(rc, GEOM_EPSILON),
            potential=ctx.potential,
            target_ids=ids0,
            candidate_ids=ids_all,
            atom_types=ctx.atom_types,
            backend=ctx.backend,
            prefer_marked_dirty=True,
        )
        if f is not None:
            return f
    return forces_on_targets_celllist_compact(
        ctx.r, ctx.box, ctx.potential, ctx.cutoff, ids0, cell, atom_types=ctx.atom_types
    )


def _forces_3d_post(
    ctx: _TDLocalCtx,
    ids0: np.ndarray,
    ids_all: np.ndarray,
    cell: Any,
    rc: float,
    candidate_ids: np.ndarray,
) -> np.ndarray:
    """Force computation for async 3D path (velocity-finish half)."""
    if ctx.many_body:
        return _forces_many_body_targets(ctx, ids0, candidate_ids, rc)
    if ctx.use_gpu_pair:
        f2 = forces_on_targets_celllist_backend(
            r=ctx.r,
            box=ctx.box,
            cutoff=ctx.cutoff,
            rc=max(rc, GEOM_EPSILON),
            potential=ctx.potential,
            target_ids=ids0,
            candidate_ids=ids_all,
            atom_types=ctx.atom_types,
            backend=ctx.backend,
            prefer_marked_dirty=True,
        )
        if f2 is not None:
            return f2
    cell = build_cell_list(ctx.r, ids_all, ctx.box, rc=max(rc, GEOM_EPSILON))
    return forces_on_targets_celllist_compact(
        ctx.r, ctx.box, ctx.potential, ctx.cutoff, ids0, cell, atom_types=ctx.atom_types
    )


# ---------------------------------------------------------------------------
# Public entry point (signature preserved for backward compatibility)
# ---------------------------------------------------------------------------


def _run_td_local_legacy(
    r: np.ndarray,
    v: np.ndarray,
    mass: Union[float, np.ndarray],
    box: float,
    potential,
    dt: float,
    cutoff: float,
    n_steps: int,
    observer=None,
    observer_every: int = 0,
    trace=None,
    atom_types: np.ndarray | None = None,
    chaos_mode: bool = False,
    chaos_seed: int = 12345,
    chaos_delay_prob: float = 0.0,
    cell_size: float = 1.0,
    zones_total: int = 1,
    zone_cells_w: int = 1,
    zone_cells_s: int = 1,
    zone_cells_pattern=None,
    traversal: str = "forward",
    buffer_k: float = 1.2,
    skin_from_buffer: bool = True,
    use_verlet: bool = True,
    verlet_k_steps: int = 20,
    decomposition: str = "1d",
    sync_mode: bool = False,
    zones_nx: int = 1,
    zones_ny: int = 1,
    zones_nz: int = 1,
    strict_min_zone_width: bool = False,
    ensemble_kind: str = "nve",
    thermostat: object | None = None,
    barostat: object | None = None,
    device: str = "cpu",
):
    """TD-local (single-process) reference implementation.

    decomposition:
      - "1d": legacy slab zones along Z (default; fully optimized path with local bins)
      - "3d": 3D block zones (correctness-first path; uses global cell-list for forces)
    sync_mode:
      - False (default): asynchronous zone-by-zone update (TD-style)
      - True: synchronous snapshot update (verification-oriented)
    """
    rng = np.random.default_rng(int(chaos_seed)) if chaos_mode else None
    mass_scalar, mass_arr, inv_mass = normalize_mass(mass, n_atoms=r.shape[0])
    atom_types = normalize_atom_types(atom_types, n_atoms=r.shape[0])
    backend = resolve_backend(device)
    use_gpu_pair = (backend.device == "cuda") and supports_pair_gpu(potential)
    ensemble = build_ensemble_spec(
        kind=ensemble_kind, thermostat=thermostat, barostat=barostat, source="td_local"
    )
    many_body = hasattr(potential, "forces_energy_virial")
    many_body_force_scope = td_local_many_body_force_scope(
        potential,
        sync_mode=bool(sync_mode),
        decomposition=str(decomposition),
        device=str(backend.device),
    )
    wave_batch_diagnostics = _WaveBatchDiagnosticsAccumulator(
        effective_device=str(backend.device),
        decomposition=str(decomposition).strip().lower(),
        sync_mode=bool(sync_mode),
        eligible=bool(
            (not bool(sync_mode))
            and str(decomposition).strip().lower() == "1d"
            and str(backend.device) == "cuda"
            and (bool(use_gpu_pair) or bool(many_body))
        ),
        many_body=bool(many_body),
        pair_gpu=bool(use_gpu_pair),
        steps_total=int(n_steps),
    )
    _set_last_td_local_wave_batch_diagnostics(wave_batch_diagnostics.finalize())

    ctx = _TDLocalCtx(
        r=r,
        v=v,
        box=box,
        potential=potential,
        dt=dt,
        cutoff=cutoff,
        n_steps=n_steps,
        atom_types=atom_types,
        mass=mass,
        mass_scalar=mass_scalar,
        mass_arr=mass_arr,
        inv_mass=inv_mass,
        backend=backend,
        use_gpu_pair=use_gpu_pair,
        ensemble=ensemble,
        many_body=many_body,
        many_body_force_scope=many_body_force_scope,
        rc_full=max(float(cutoff), GEOM_EPSILON),
        wave_batch_diagnostics=wave_batch_diagnostics,
        observer=observer,
        observer_every=observer_every,
        accepts_box=observer_accepts_box(observer),
        trace=trace,
        rng=rng,
        cell_size=cell_size,
        zones_total=zones_total,
        zone_cells_w=zone_cells_w,
        zone_cells_s=zone_cells_s,
        zone_cells_pattern=zone_cells_pattern,
        traversal=traversal,
        buffer_k=buffer_k,
        skin_from_buffer=skin_from_buffer,
        use_verlet=use_verlet,
        verlet_k_steps=verlet_k_steps,
        strict_min_zone_width=strict_min_zone_width,
        zones_nx=zones_nx,
        zones_ny=zones_ny,
        zones_nz=zones_nz,
        ensemble_kind=ensemble_kind,
    )

    try:
        # Dispatch to the appropriate execution path
        if sync_mode:
            if many_body or decomposition.lower() == "3d":
                _run_sync_global(ctx)
            else:
                _run_sync_1d_zones(ctx)
        elif decomposition.lower() == "3d":
            _run_async_3d(ctx)
        else:
            _run_async_1d(ctx)
    finally:
        _set_last_td_local_wave_batch_diagnostics(wave_batch_diagnostics.finalize())


def describe_many_body_force_scope(
    potential: object,
    *,
    sync_mode: bool = False,
    decomposition: str = "1d",
    device: str = "cpu",
) -> dict[str, object] | None:
    """Describe the current td_local many-body force scope without executing the runtime."""
    scope = td_local_many_body_force_scope(
        potential,
        sync_mode=bool(sync_mode),
        decomposition=str(decomposition),
        device=str(device),
    )
    if scope is None:
        return None
    return scope.as_dict()


def describe_td_local_wave_batch_contract(
    *,
    sync_mode: bool = False,
    decomposition: str = "1d",
    device: str = "cpu",
) -> dict[str, object] | None:
    if bool(sync_mode):
        return None
    if str(decomposition).strip().lower() != "1d":
        return None
    if str(device).strip().lower() != "cuda":
        return None
    return {
        "version": TD_LOCAL_WAVE_BATCH_CONTRACT_VERSION,
        "mode": "td_local.async_1d.cuda",
        "wavefront_contract_version": WAVEFRONT_1D_CONTRACT_VERSION,
        "diagnostics_contract_version": TD_LOCAL_WAVE_BATCH_DIAGNOSTICS_VERSION,
        "pre_force_batching": True,
        "post_force_batching": False,
        "state_progression": "sequential_per_zone",
        "formal_core_w_leq_1": True,
    }


def run_td_local(
    r: np.ndarray,
    v: np.ndarray,
    mass: Union[float, np.ndarray],
    box: float,
    potential,
    dt: float,
    cutoff: float,
    n_steps: int,
    observer=None,
    observer_every: int = 0,
    trace=None,
    *,
    config: TDLocalRunConfig | None = None,
    **legacy_kwargs: Any,
):
    """TD-local public entry point with compact config object.

    Backward compatibility:
      - legacy keyword options (atom_types, zones_total, etc.) are still accepted;
      - when both ``config`` and legacy kwargs are provided, raises ``TypeError``.
    """
    if config is not None and legacy_kwargs:
        keys = ", ".join(sorted(legacy_kwargs.keys()))
        raise TypeError(
            f"run_td_local received both config and legacy keyword options ({keys}); use one style"
        )
    cfg = config if config is not None else TDLocalRunConfig.from_legacy_kwargs(legacy_kwargs)
    return _run_td_local_legacy(
        r=r,
        v=v,
        mass=mass,
        box=box,
        potential=potential,
        dt=dt,
        cutoff=cutoff,
        n_steps=n_steps,
        observer=observer,
        observer_every=observer_every,
        trace=trace,
        atom_types=cfg.atom_types,
        chaos_mode=cfg.chaos_mode,
        chaos_seed=cfg.chaos_seed,
        chaos_delay_prob=cfg.chaos_delay_prob,
        cell_size=cfg.cell_size,
        zones_total=cfg.zones_total,
        zone_cells_w=cfg.zone_cells_w,
        zone_cells_s=cfg.zone_cells_s,
        zone_cells_pattern=cfg.zone_cells_pattern,
        traversal=cfg.traversal,
        buffer_k=cfg.buffer_k,
        skin_from_buffer=cfg.skin_from_buffer,
        use_verlet=cfg.use_verlet,
        verlet_k_steps=cfg.verlet_k_steps,
        decomposition=cfg.decomposition,
        sync_mode=cfg.sync_mode,
        zones_nx=cfg.zones_nx,
        zones_ny=cfg.zones_ny,
        zones_nz=cfg.zones_nz,
        strict_min_zone_width=cfg.strict_min_zone_width,
        ensemble_kind=cfg.ensemble_kind,
        thermostat=cfg.thermostat,
        barostat=cfg.barostat,
        device=cfg.device,
    )
