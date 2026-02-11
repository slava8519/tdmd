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

from dataclasses import dataclass
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
    supports_pair_gpu,
)
from .integrator import vv_finish_velocities, vv_update_positions
from .observer import emit_observer, observer_accepts_box
from .run_configs import TDLocalRunConfig
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
    rc_full: float

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
        f1 = ctx.forces_full(ctx.r)
        ctx.v[:] = v_half + 0.5 * ctx.dt * ctx.accel(f1, ids_all)
        ctx.box, _lam_b = ctx.apply_ensemble(step, atom_box=ctx.box)
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
                    f = _forces_3d(ctx, ids0, ids_all, cell, rc)
                    vv_update_positions(ctx.r, ctx.v, ctx.mass, ctx.dt, ctx.box, ids0, f)
                    processed[ids0] = True
                    assign_atoms_to_zones_3d(ctx.r, layout3)

                    f2 = _forces_3d_post(ctx, ids0, ids_all, cell, rc)
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
                    if ctx.many_body:
                        f = ctx.forces_full(ctx.r)[ids0]
                    else:
                        f = _forces_on_zone_1d_async(
                            ctx, ids0, candidate_ids, cache, zid, rc, skin_global, step, z0p, z1p
                        )
                    vv_update_positions(ctx.r, ctx.v, ctx.mass, ctx.dt, ctx.box, ids0, f)
                    assign_atoms_to_zones(ctx.r, zones, ctx.box)

                    # ---- velocity finish (VV half-step 2) ----
                    cand2 = []
                    for nzid in pzids:
                        nz = zones[nzid]
                        if nz.atom_ids.size:
                            cand2.append(nz.atom_ids)
                    candidate_ids2 = np.concatenate(cand2) if cand2 else np.empty((0,), np.int32)
                    if ctx.many_body:
                        f2 = ctx.forces_full(ctx.r)[ids0]
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
        if ctx.ensemble_kind == "npt":
            _scale_zones_1d(zones, lam_b, ctx.cutoff)

        if ctx.observer is not None and ctx.observer_every and (step % ctx.observer_every == 0):
            ctx.emit_observer(step)


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


def _forces_3d(
    ctx: _TDLocalCtx,
    ids0: np.ndarray,
    ids_all: np.ndarray,
    cell: Any,
    rc: float,
) -> np.ndarray:
    """Force computation for async 3D path (position-update half)."""
    if ctx.many_body:
        return ctx.forces_full(ctx.r)[ids0]
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
) -> np.ndarray:
    """Force computation for async 3D path (velocity-finish half)."""
    if ctx.many_body:
        return ctx.forces_full(ctx.r)[ids0]
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
        rc_full=max(float(cutoff), GEOM_EPSILON),
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
