from __future__ import annotations
from typing import Optional, Union
import inspect
import numpy as np

from .zones import (
    ZoneType, ZoneLayout1DCells, assign_atoms_to_zones, compute_zone_buffer_skin, zones_overlapping_range_pbc,
    ZoneLayout3DBlocks, assign_atoms_to_zones_3d, zones_overlapping_aabb_pbc,
)
from .integrator import vv_update_positions, vv_finish_velocities
from .zone_bins_localz import PersistentZoneLocalZBinsCache
from .forces_cells import forces_on_targets_zonecells, forces_on_targets_celllist_compact
from .celllist import build_cell_list
from .backend import resolve_backend
from .ensembles import apply_ensemble_step, build_ensemble_spec
from .forces_gpu import (
    forces_on_targets_celllist_backend,
    forces_on_targets_pair_backend,
    supports_pair_gpu,
)
from .potentials import EAMAlloyPotential

def run_td_local(r: np.ndarray, v: np.ndarray, mass: Union[float, np.ndarray], box: float, potential,
                 dt: float, cutoff: float, n_steps: int,
                 observer=None, observer_every: int = 0,
                 trace=None,
                 atom_types: np.ndarray | None = None,
                 chaos_mode: bool = False, chaos_seed: int = 12345, chaos_delay_prob: float = 0.0,
                 cell_size: float = 1.0, zones_total: int = 1, zone_cells_w: int = 1, zone_cells_s: int = 1, zone_cells_pattern=None,
                 traversal: str = "forward",
                 buffer_k: float = 1.2, skin_from_buffer: bool = True,
                 use_verlet: bool = True, verlet_k_steps: int = 20,
                 decomposition: str = "1d",
                 sync_mode: bool = False,
                 zones_nx: int = 1, zones_ny: int = 1, zones_nz: int = 1,
                 strict_min_zone_width: bool = False,
                 ensemble_kind: str = "nve",
                 thermostat: object | None = None,
                 barostat: object | None = None,
                 device: str = "cpu"):
    """TD-local (single-process) reference implementation.

    decomposition:
      - "1d": legacy slab zones along Z (default; fully optimized path with local bins)
      - "3d": 3D block zones (correctness-first path; uses global cell-list for forces)
    sync_mode:
      - False (default): asynchronous zone-by-zone update (TD-style)
      - True: synchronous snapshot update (verification-oriented)
    """
    rng = np.random.default_rng(int(chaos_seed)) if chaos_mode else None
    if np.isscalar(mass):
        mass_scalar = float(mass)
        if mass_scalar <= 0.0:
            raise ValueError("mass must be positive")
        mass_arr = None
        inv_mass = 1.0 / mass_scalar
    else:
        mass_arr = np.asarray(mass, dtype=float)
        if mass_arr.ndim != 1 or mass_arr.shape[0] != r.shape[0]:
            raise ValueError("mass array must have shape (N,)")
        if np.any(mass_arr <= 0.0):
            raise ValueError("all masses must be positive")
        mass_scalar = None
        inv_mass = None

    def _accel(f: np.ndarray, ids: Optional[np.ndarray] = None) -> np.ndarray:
        if mass_arr is None:
            return f * inv_mass
        if ids is None:
            return f / mass_arr[:, None]
        return f / mass_arr[ids][:, None]

    if atom_types is None:
        atom_types = np.ones(r.shape[0], dtype=np.int32)
    else:
        atom_types = np.asarray(atom_types, dtype=np.int32)
        if atom_types.ndim != 1 or atom_types.shape[0] != r.shape[0]:
            raise ValueError("atom_types must have shape (N,)")
    backend = resolve_backend(device)
    use_gpu_pair = (backend.device == "cuda") and supports_pair_gpu(potential)
    ensemble = build_ensemble_spec(
        kind=ensemble_kind,
        thermostat=thermostat,
        barostat=barostat,
        source="td_local",
    )
    observer_accepts_box = False
    if observer is not None:
        try:
            sig = inspect.signature(observer)
            params = list(sig.parameters.values())
            observer_accepts_box = (
                any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
                or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
                or len(params) >= 4
            )
        except (TypeError, ValueError):
            observer_accepts_box = False

    def _emit_observer(step: int) -> None:
        if observer is None or not observer_every:
            return
        if observer_accepts_box:
            observer(int(step), r, v, float(box))
        else:
            observer(int(step), r, v)

    many_body = hasattr(potential, "forces_energy_virial")

    def _apply_ensemble(step: int, *, atom_box: float) -> tuple[float, float]:
        new_box, _lam_t, lam_b = apply_ensemble_step(
            step=int(step),
            ensemble=ensemble,
            r=r,
            v=v,
            mass=(mass_scalar if mass_arr is None else mass_arr),
            box=float(atom_box),
            potential=potential,
            cutoff=cutoff,
            atom_types=atom_types,
            dt=dt,
        )
        return float(new_box), float(lam_b)

    def _scale_zones_1d(zones, lam: float) -> None:
        if abs(float(lam) - 1.0) <= 1e-15:
            return
        for z in zones:
            z.z0 = float(z.z0) * float(lam)
            z.z1 = float(z.z1) * float(lam)
        widths = [float(z.z1 - z.z0) for z in zones if getattr(z, "n_cells", 0) > 0 and z.z1 > z.z0]
        if widths and (min(widths) + 1e-12 < float(cutoff)):
            raise ValueError("NPT scaling violated zone width >= cutoff in td_local 1D layout")

    def _scale_layout3(layout3, lam: float, new_box: float) -> None:
        if abs(float(lam) - 1.0) <= 1e-15:
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
        if widths and (min(widths) + 1e-12 < float(cutoff)):
            raise ValueError("NPT scaling violated zone width >= cutoff in td_local 3D layout")

    def _forces_full(rr: np.ndarray) -> np.ndarray:
        if use_gpu_pair:
            ids_all = np.arange(rr.shape[0], dtype=np.int32)
            if isinstance(potential, EAMAlloyPotential):
                f_gpu = forces_on_targets_pair_backend(
                    r=rr,
                    box=box,
                    cutoff=cutoff,
                    potential=potential,
                    target_ids=ids_all,
                    candidate_ids=ids_all,
                    atom_types=atom_types,
                    backend=backend,
                )
                if f_gpu is not None:
                    return np.asarray(f_gpu, dtype=float)
            else:
                f_gpu = forces_on_targets_celllist_backend(
                    r=rr,
                    box=box,
                    cutoff=cutoff,
                    rc=max(cutoff, 1e-12),
                    potential=potential,
                    target_ids=ids_all,
                    candidate_ids=ids_all,
                    atom_types=atom_types,
                    backend=backend,
                )
                if f_gpu is not None:
                    return np.asarray(f_gpu, dtype=float)
            f_gpu2 = forces_on_targets_pair_backend(
                r=rr,
                box=box,
                cutoff=cutoff,
                potential=potential,
                target_ids=ids_all,
                candidate_ids=ids_all,
                atom_types=atom_types,
                backend=backend,
            )
            if f_gpu2 is not None:
                return np.asarray(f_gpu2, dtype=float)
        if many_body:
            f_all, _pe, _w = potential.forces_energy_virial(rr, box, cutoff, atom_types)
            return np.asarray(f_all, dtype=float)
        ids_all = np.arange(rr.shape[0], dtype=np.int32)
        cell = build_cell_list(rr, ids_all, box, rc=max(cutoff, 1e-12))
        return forces_on_targets_celllist_compact(
            rr, box, potential, cutoff, ids_all, cell, atom_types=atom_types
        )

    if sync_mode:
        # Synchronous snapshot update for verification: all forces computed from a consistent
        # time layer, then positions/velocities updated once per global step.
        if many_body:
            ids_all = np.arange(r.shape[0], dtype=np.int32)
            if observer is not None and observer_every:
                _emit_observer(0)
            for step in range(1, n_steps + 1):
                f0 = _forces_full(r)
                v_half = v + 0.5 * dt * _accel(f0, ids_all)
                r[:] = (r + dt * v_half) % box
                f1 = _forces_full(r)
                v[:] = v_half + 0.5 * dt * _accel(f1, ids_all)
                box, _lam_b = _apply_ensemble(step, atom_box=box)
                if observer is not None and observer_every and (step % observer_every == 0):
                    _emit_observer(step)
            return

        if decomposition.lower() == "3d":
            ids_all = np.arange(r.shape[0], dtype=np.int32)
            if observer is not None and observer_every:
                _emit_observer(0)
            for step in range(1, n_steps+1):
                f0 = _forces_full(r)
                v_half = v + 0.5 * dt * _accel(f0, ids_all)
                r[:] = (r + dt*v_half) % box
                f1 = _forces_full(r)
                v[:] = v_half + 0.5 * dt * _accel(f1, ids_all)
                box, _lam_b = _apply_ensemble(step, atom_box=box)
                if observer is not None and observer_every and (step % observer_every == 0):
                    _emit_observer(step)
            return

        layout = ZoneLayout1DCells(
            box=box, cell_size=cell_size, zones_total=zones_total,
            pattern_cells=zone_cells_pattern, zone_cells_w=zone_cells_w, zone_cells_s=zone_cells_s,
            min_zone_width=float(cutoff),
            strict_min_width=bool(strict_min_zone_width),
        )
        zones = layout.build()
        if traversal == "backward":
            order = list(range(zones_total-1, -1, -1))
        else:
            order = list(range(zones_total))
        cache = PersistentZoneLocalZBinsCache()
        if observer is not None and observer_every:
            _emit_observer(0)
        for step in range(1, n_steps+1):
            assign_atoms_to_zones(r, zones, box)
            for z in zones:
                z.ztype = ZoneType.D if z.atom_ids.size else ZoneType.F

            rc = float(cutoff)
            f0 = np.zeros_like(r)
            for zid in order:
                z = zones[zid]
                if z.atom_ids.size == 0:
                    continue
                z0p = z.z0 - cutoff
                z1p = z.z1 + cutoff
                pzids = zones_overlapping_range_pbc(z0p, z1p, box, zones)
                cand=[]
                for nzid in pzids:
                    nz = zones[nzid]
                    if nz.atom_ids.size:
                        cand.append(nz.atom_ids)
                candidate_ids = np.concatenate(cand) if cand else np.empty((0,), np.int32)
                if candidate_ids.size:
                    if use_gpu_pair:
                        f_zone = forces_on_targets_pair_backend(
                            r=r,
                            box=box,
                            cutoff=cutoff,
                            potential=potential,
                            target_ids=z.atom_ids,
                            candidate_ids=candidate_ids,
                            atom_types=atom_types,
                            backend=backend,
                        )
                        if f_zone is None:
                            f_zone = np.zeros((z.atom_ids.size, 3), dtype=np.float64)
                    else:
                        zc = cache.get(zid, r, box, candidate_ids, rc=rc,
                                       skin_global=0.0,
                                       step=2*step, verlet_k_steps=1,
                                       z0=z0p, z1=z1p)
                        f_zone = forces_on_targets_zonecells(
                            r, box, potential, cutoff, z.atom_ids, zc, atom_types=atom_types
                        )
                    f0[z.atom_ids] = f_zone

            v_half = v + 0.5 * dt * _accel(f0)
            r[:] = (r + dt*v_half) % box

            assign_atoms_to_zones(r, zones, box)
            for z in zones:
                z.ztype = ZoneType.D if z.atom_ids.size else ZoneType.F

            f1 = np.zeros_like(r)
            for zid in order:
                z = zones[zid]
                if z.atom_ids.size == 0:
                    continue
                z0p = z.z0 - cutoff
                z1p = z.z1 + cutoff
                pzids = zones_overlapping_range_pbc(z0p, z1p, box, zones)
                cand=[]
                for nzid in pzids:
                    nz = zones[nzid]
                    if nz.atom_ids.size:
                        cand.append(nz.atom_ids)
                candidate_ids = np.concatenate(cand) if cand else np.empty((0,), np.int32)
                if candidate_ids.size:
                    if use_gpu_pair:
                        f_zone = forces_on_targets_pair_backend(
                            r=r,
                            box=box,
                            cutoff=cutoff,
                            potential=potential,
                            target_ids=z.atom_ids,
                            candidate_ids=candidate_ids,
                            atom_types=atom_types,
                            backend=backend,
                        )
                        if f_zone is None:
                            f_zone = np.zeros((z.atom_ids.size, 3), dtype=np.float64)
                    else:
                        zc = cache.get(zid, r, box, candidate_ids, rc=rc,
                                       skin_global=0.0,
                                       step=2*step+1, verlet_k_steps=1,
                                       z0=z0p, z1=z1p)
                        f_zone = forces_on_targets_zonecells(
                            r, box, potential, cutoff, z.atom_ids, zc, atom_types=atom_types
                        )
                    f1[z.atom_ids] = f_zone

            v[:] = v_half + 0.5 * dt * _accel(f1)
            box, lam_b = _apply_ensemble(step, atom_box=box)
            if ensemble.kind == "npt":
                _scale_zones_1d(zones, lam_b)
            if observer is not None and observer_every and (step % observer_every == 0):
                _emit_observer(step)
        return

    if decomposition.lower() == "3d":
        layout3 = ZoneLayout3DBlocks.build(box=box, nx=int(zones_nx), ny=int(zones_ny), nz=int(zones_nz))
        zones3 = layout3.zones
        assign_atoms_to_zones_3d(r, layout3)
        for z in zones3:
            if z.atom_ids.size:
                z.ztype = ZoneType.D

        order = list(range(len(zones3)))
        if traversal == "backward":
            order = list(range(len(zones3)-1, -1, -1))

        if observer is not None and observer_every:
            _emit_observer(0)

        for step in range(1, n_steps+1):
            if chaos_mode:
                rng.shuffle(order)

            for z in zones3:
                z.ztype = ZoneType.D if z.atom_ids.size else ZoneType.F
            processed = np.zeros(r.shape[0], dtype=bool)

            # compute global skin as safety envelope (same idea as 1D path)
            skin_global = 0.0
            for z in zones3:
                b, skin = compute_zone_buffer_skin(v, z.atom_ids, dt, buffer_k, skin_from_buffer=skin_from_buffer)
                z.buffer = b; z.skin = skin
                skin_global = max(skin_global, skin)
            rc = cutoff + skin_global

            # rebuild global cell list for this step (correctness-first)
            ids_all = np.arange(r.shape[0], dtype=np.int32)
            cell = build_cell_list(r, ids_all, box, rc=max(rc, 1e-12))

            for zid in order:
                z = zones3[zid]
                if z.ztype in (ZoneType.D, ZoneType.P) and z.atom_ids.size:
                    ids0 = z.atom_ids[~processed[z.atom_ids]]
                    if ids0.size == 0:
                        z.ztype = ZoneType.S
                        continue
                    state_before = z.ztype
                    z.ztype = ZoneType.W
                    if trace is not None:
                        trace.log(step_id=int(step), zone_id=int(zid), event="START_COMPUTE",
                                  state_before=state_before, state_after=z.ztype,
                                  halo_ids_count=0, migration_count=0, lag=0, invariant_flags="")

                    lo = z.lo - cutoff
                    hi = z.hi + cutoff
                    deps = zones_overlapping_aabb_pbc(lo, hi, box, layout3)
                    # mark deps as P if needed
                    cand=[]
                    for did in deps:
                        dz = zones3[did]
                        if dz.atom_ids.size:
                            if dz.ztype == ZoneType.D and did != zid:
                                dz.ztype = ZoneType.P
                            cand.append(dz.atom_ids)
                    candidate_ids = np.concatenate(cand) if cand else np.empty((0,), np.int32)

                    if candidate_ids.size or many_body:
                        if many_body:
                            f = _forces_full(r)[ids0]
                        else:
                            if use_gpu_pair:
                                f = forces_on_targets_celllist_backend(
                                    r=r,
                                    box=box,
                                    cutoff=cutoff,
                                    rc=max(rc, 1e-12),
                                    potential=potential,
                                    target_ids=ids0,
                                    candidate_ids=ids_all,
                                    atom_types=atom_types,
                                    backend=backend,
                                )
                                if f is None:
                                    f = forces_on_targets_celllist_compact(
                                        r, box, potential, cutoff, ids0, cell, atom_types=atom_types
                                    )
                            else:
                                f = forces_on_targets_celllist_compact(
                                    r, box, potential, cutoff, ids0, cell, atom_types=atom_types
                                )
                        vv_update_positions(r, v, mass, dt, box, ids0, f)
                        processed[ids0] = True
                        assign_atoms_to_zones_3d(r, layout3)

                        # refresh cell list (positions changed) for velocity finish
                        if many_body:
                            f2 = _forces_full(r)[ids0]
                        else:
                            if use_gpu_pair:
                                f2 = forces_on_targets_celllist_backend(
                                    r=r,
                                    box=box,
                                    cutoff=cutoff,
                                    rc=max(rc, 1e-12),
                                    potential=potential,
                                    target_ids=ids0,
                                    candidate_ids=ids_all,
                                    atom_types=atom_types,
                                    backend=backend,
                                )
                                if f2 is None:
                                    cell = build_cell_list(r, ids_all, box, rc=max(rc, 1e-12))
                                    f2 = forces_on_targets_celllist_compact(
                                        r, box, potential, cutoff, ids0, cell, atom_types=atom_types
                                    )
                            else:
                                cell = build_cell_list(r, ids_all, box, rc=max(rc, 1e-12))
                                f2 = forces_on_targets_celllist_compact(
                                    r, box, potential, cutoff, ids0, cell, atom_types=atom_types
                                )
                        vv_finish_velocities(v, mass, dt, ids0, f2)

                    state_before = z.ztype
                    z.ztype = ZoneType.S
                    if trace is not None:
                        trace.log(step_id=int(step), zone_id=int(zid), event="FINISH_COMPUTE",
                                  state_before=state_before, state_after=z.ztype,
                                  halo_ids_count=0, migration_count=0, lag=0, invariant_flags="")

            box, lam_b = _apply_ensemble(step, atom_box=box)
            if ensemble.kind == "npt":
                _scale_layout3(layout3, lam_b, box)

            if observer is not None and observer_every and (step % observer_every == 0):
                _emit_observer(step)

        return  # end 3D

    # ---------------------
    # Legacy 1D slab mode
    # ---------------------
    layout = ZoneLayout1DCells(
        box=box, cell_size=cell_size, zones_total=zones_total,
        pattern_cells=zone_cells_pattern, zone_cells_w=zone_cells_w, zone_cells_s=zone_cells_s,
        min_zone_width=float(cutoff),
        strict_min_width=bool(strict_min_zone_width),
    )
    zones = layout.build()
    assign_atoms_to_zones(r, zones, box)
    for z in zones:
        z.ztype = ZoneType.D

    if traversal == "backward":
        order = list(range(zones_total-1,-1,-1))
    else:
        order = list(range(zones_total))

    cache = PersistentZoneLocalZBinsCache()

    if observer is not None and observer_every:
        _emit_observer(0)
    for step in range(1, n_steps+1):
        if chaos_mode:
            rng.shuffle(order)
        for z in zones:
            z.ztype = ZoneType.D if z.atom_ids.size else ZoneType.F
        processed = np.zeros(r.shape[0], dtype=bool)
        skin_global = 0.0
        for z in zones:
            b, skin = compute_zone_buffer_skin(v, z.atom_ids, dt, buffer_k, skin_from_buffer=skin_from_buffer)
            z.buffer=b; z.skin=skin
            skin_global = max(skin_global, skin)
        rc = cutoff + skin_global

        for zid in order:
            z = zones[zid]
            if z.ztype in (ZoneType.D, ZoneType.P) and z.atom_ids.size:
                ids0 = z.atom_ids[~processed[z.atom_ids]]
                if ids0.size == 0:
                    z.ztype = ZoneType.S
                    continue
                state_before = z.ztype
                z.ztype = ZoneType.W
                if trace is not None:
                    trace.log(step_id=int(step), zone_id=int(zid), event="START_COMPUTE",
                              state_before=state_before, state_after=z.ztype,
                              halo_ids_count=0, migration_count=0, lag=0, invariant_flags="")
                z0p = z.z0 - cutoff
                z1p = z.z1 + cutoff
                pzids = zones_overlapping_range_pbc(z0p, z1p, box, zones)
                cand=[]
                for nzid in pzids:
                    nz = zones[nzid]
                    if nz.atom_ids.size:
                        if nz.ztype == ZoneType.D and nzid != zid:
                            nz.ztype = ZoneType.P
                        cand.append(nz.atom_ids)
                candidate_ids = np.concatenate(cand) if cand else np.empty((0,), np.int32)

                if candidate_ids.size or many_body:
                    if many_body:
                        f = _forces_full(r)[ids0]
                    else:
                        if use_gpu_pair:
                            f = forces_on_targets_pair_backend(
                                r=r,
                                box=box,
                                cutoff=cutoff,
                                potential=potential,
                                target_ids=ids0,
                                candidate_ids=candidate_ids,
                                atom_types=atom_types,
                                backend=backend,
                            )
                            if f is None:
                                zc = cache.get(zid, r, box, candidate_ids, rc=rc,
                                               skin_global=(skin_global if use_verlet else 0.0),
                                               step=step, verlet_k_steps=(verlet_k_steps if use_verlet else 1),
                                               z0=z0p, z1=z1p)
                                f = forces_on_targets_zonecells(
                                    r, box, potential, cutoff, ids0, zc, atom_types=atom_types
                                )
                        else:
                            zc = cache.get(zid, r, box, candidate_ids, rc=rc,
                                           skin_global=(skin_global if use_verlet else 0.0),
                                           step=step, verlet_k_steps=(verlet_k_steps if use_verlet else 1),
                                           z0=z0p, z1=z1p)
                            f = forces_on_targets_zonecells(
                                r, box, potential, cutoff, ids0, zc, atom_types=atom_types
                            )
                    vv_update_positions(r, v, mass, dt, box, ids0, f)
                    assign_atoms_to_zones(r, zones, box)

                    cand2=[]
                    for nzid in pzids:
                        nz = zones[nzid]
                        if nz.atom_ids.size:
                            cand2.append(nz.atom_ids)
                    candidate_ids2 = np.concatenate(cand2) if cand2 else np.empty((0,), np.int32)
                    if many_body:
                        f2 = _forces_full(r)[ids0]
                    elif candidate_ids2.size:
                        if use_gpu_pair:
                            f2 = forces_on_targets_pair_backend(
                                r=r,
                                box=box,
                                cutoff=cutoff,
                                potential=potential,
                                target_ids=ids0,
                                candidate_ids=candidate_ids2,
                                atom_types=atom_types,
                                backend=backend,
                            )
                            if f2 is None:
                                zc2 = cache.get(zid, r, box, candidate_ids2, rc=rc,
                                                skin_global=(skin_global if use_verlet else 0.0),
                                                step=step, verlet_k_steps=(verlet_k_steps if use_verlet else 1),
                                                z0=z0p, z1=z1p)
                                f2 = forces_on_targets_zonecells(
                                    r, box, potential, cutoff, ids0, zc2, atom_types=atom_types
                                )
                        else:
                            zc2 = cache.get(zid, r, box, candidate_ids2, rc=rc,
                                            skin_global=(skin_global if use_verlet else 0.0),
                                            step=step, verlet_k_steps=(verlet_k_steps if use_verlet else 1),
                                            z0=z0p, z1=z1p)
                            f2 = forces_on_targets_zonecells(
                                r, box, potential, cutoff, ids0, zc2, atom_types=atom_types
                            )
                    else:
                        f2 = np.zeros((ids0.size, 3), dtype=np.float64)
                    vv_finish_velocities(v, mass, dt, ids0, f2)

                state_before = z.ztype
                z.ztype = ZoneType.S
                if trace is not None:
                    trace.log(step_id=int(step), zone_id=int(zid), event="FINISH_COMPUTE",
                              state_before=state_before, state_after=z.ztype,
                              halo_ids_count=0, migration_count=0, lag=0, invariant_flags="")

        box, lam_b = _apply_ensemble(step, atom_box=box)
        if ensemble.kind == "npt":
            _scale_zones_1d(zones, lam_b)

        if observer is not None and observer_every and (step % observer_every == 0):
            _emit_observer(step)
