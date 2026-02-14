from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Union

import numpy as np

try:
    import mpi4py

    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False
    from mpi4py import MPI
except Exception:
    MPI = None

from .atoms import normalize_atom_types
from .backend import cuda_device_for_local_rank, local_rank_from_env, resolve_backend
from .constants import FLOAT_EQ_ATOL, GEOM_EPSILON
from .deps_provider import ZoneGeomAABB
from .deps_provider_3d import DepsProvider3DBlock
from .ensembles import apply_ensemble_step, build_ensemble_spec
from .force_dispatch import try_gpu_forces_on_targets
from .geom_pbc import mask_in_aabb_pbc
from .output import OutputSpec, make_output_bundle
from .overlap_cells import overlap_atoms_src_in_next_zonecells
from .run_configs import TDFullMPIRunConfig
from .state import kinetic_energy, temperature_from_ke
from .td_automaton import TDAutomaton1W, ZoneRuntime
from .td_trace import TDTraceLogger, format_invariant_flags
from .zone_bins_localz import PersistentZoneLocalZBinsCache
from .zones import (
    ZoneLayout1DCells,
    ZoneType,
    assign_atoms_to_zones,
    compute_zone_buffer_skin,
    zones_overlapping_range_pbc,
)


@dataclass
class TDStats:
    rank: int
    size: int


@dataclass(frozen=True)
class _TDMPIRuntimeInit:
    batch_size: int
    comm: object
    rank: int
    size: int
    prev_rank: int
    next_rank: int
    ensemble: object
    backend: object
    cuda_aware_active: bool
    use_async_send: bool
    owns_mpi_init: bool


_WireRecord = tuple[int, int, int, np.ndarray, int]


class _GPUPotentialRefinement:
    """GPU refinement wrapper for force callbacks in TD-MPI compute path.

    Contract:
    - CPU path remains reference semantics.
    - On CUDA backend, try GPU force path first.
    - If GPU path is unavailable for a call, fall back to wrapped CPU potential.
    """

    def __init__(self, base_potential, backend):
        self._base = base_potential
        self._backend = backend

    def __getattr__(self, name: str):
        return getattr(self._base, name)

    def forces_on_targets(
        self,
        *,
        r: np.ndarray,
        box: float,
        cutoff: float,
        rc: float | None = None,
        atom_types: np.ndarray,
        target_ids: np.ndarray,
        candidate_ids: np.ndarray,
    ) -> np.ndarray:
        rc_eff = float(cutoff) if rc is None else float(rc)
        f_gpu = try_gpu_forces_on_targets(
            r=r,
            box=float(box),
            cutoff=float(cutoff),
            rc=rc_eff,
            potential=self._base,
            target_ids=np.asarray(target_ids, dtype=np.int32),
            candidate_ids=np.asarray(candidate_ids, dtype=np.int32),
            atom_types=np.asarray(atom_types, dtype=np.int32),
            backend=self._backend,
        )
        if f_gpu is not None:
            return np.asarray(f_gpu, dtype=float)
        return self._base.forces_on_targets(
            r=r,
            box=float(box),
            cutoff=float(cutoff),
            rc=rc_eff,
            atom_types=np.asarray(atom_types, dtype=np.int32),
            target_ids=np.asarray(target_ids, dtype=np.int32),
            candidate_ids=np.asarray(candidate_ids, dtype=np.int32),
        )


def _wrap_potential_for_gpu_refinement(*, potential, backend):
    if str(getattr(backend, "device", "cpu")) != "cuda":
        return potential
    if not hasattr(potential, "forces_on_targets"):
        return potential
    return _GPUPotentialRefinement(potential, backend)


@dataclass
class _TDMPICommContext:
    comm: object
    prev_rank: int
    next_rank: int
    use_async_send: bool
    batch_size: int
    box: float
    cutoff: float
    deps_provider_mode: str
    static_rr: bool
    debug_invariants: bool
    max_step_lag: int
    max_pending_delta_atoms: int
    r: np.ndarray
    v: np.ndarray
    zones: list[ZoneRuntime]
    zones_total: int
    autom: TDAutomaton1W
    holder_map: list[int]
    holder_ver: list[int]
    req_reply_outbox: dict[int, list[_WireRecord]]
    holder_reply_outbox: dict[int, list[_WireRecord]]
    pending_deltas: dict[tuple[int, int, int], list[np.ndarray]]
    pending_atoms: int
    geom_aabb_fn: Callable[[int], ZoneGeomAABB]
    deps_zone_ids_fn: Callable[[int], list[int]]
    overlap_set_for_send_fn: Callable[[int, int, float, int], set[int]]
    trace_event_fn: Callable[..., None]
    lag_value_fn: Callable[[int], int]
    set_holder_fn: Callable[[int, int], None]
    validate_halo_geometry_fn: Callable[[int, np.ndarray], None]
    pending_send_reqs: list[object]
    pending_send_keepalive: list[object]
    pending_send_post_ts: float | None


def _init_td_full_mpi_runtime(
    *,
    device: str,
    ensemble_kind: str,
    thermostat: object | None,
    barostat: object | None,
    cuda_aware_mpi: bool,
    comm_overlap_isend: bool,
    batch_size: int,
) -> _TDMPIRuntimeInit:
    if MPI is None:
        raise RuntimeError("mpi4py required")
    owns_mpi_init = False
    if not MPI.Is_initialized():
        MPI.Init()
        owns_mpi_init = True
    batch = int(batch_size)
    if batch < 1:
        raise ValueError("batch_size/time_block_k must be >= 1")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    prev_rank = (rank - 1) % size
    next_rank = (rank + 1) % size

    ensemble = build_ensemble_spec(
        kind=ensemble_kind,
        thermostat=thermostat,
        barostat=barostat,
        source="td_full_mpi",
    )
    if ensemble.kind == "npt" and int(size) > 1:
        raise ValueError(
            "td_full_mpi NPT currently requires single-rank launch (MPI size=1) to avoid "
            "cross-rank barostat synchronization/barrier semantics"
        )

    backend = resolve_backend(str(device).strip().lower() or "auto")
    if backend.device == "cuda":
        cp = backend.xp
        local_rank = local_rank_from_env()
        try:
            ndev = int(cp.cuda.runtime.getDeviceCount())
            if ndev > 0:
                dev = cuda_device_for_local_rank(local_rank, ndev)
                cp.cuda.Device(dev).use()
                print(
                    f"[backend rank={rank}] device=cuda local_rank={local_rank} cuda_dev={dev}",
                    flush=True,
                )
        except (RuntimeError, AttributeError) as exc:
            print(
                f"[backend rank={rank}] cuda setup failed ({exc}); using host staging", flush=True
            )
    else:
        reason = f" ({backend.reason})" if backend.reason else ""
        print(f"[backend rank={rank}] device=cpu{reason}", flush=True)

    cuda_aware_active = bool(cuda_aware_mpi and (backend.device == "cuda"))
    use_async_send = bool(comm_overlap_isend or cuda_aware_active)
    if rank == 0 and bool(cuda_aware_mpi) and not cuda_aware_active:
        print(
            "[backend] cuda_aware_mpi requested but inactive (requires CUDA backend); using host-staging",
            flush=True,
        )

    return _TDMPIRuntimeInit(
        batch_size=batch,
        comm=comm,
        rank=rank,
        size=size,
        prev_rank=prev_rank,
        next_rank=next_rank,
        ensemble=ensemble,
        backend=backend,
        cuda_aware_active=cuda_aware_active,
        use_async_send=use_async_send,
        owns_mpi_init=owns_mpi_init,
    )


def _init_td_trace(*, trace_enabled: bool, trace_path: str, rank: int, size: int):
    trace = None
    if trace_enabled:
        trace_file = trace_path
        if size > 1:
            base, ext = os.path.splitext(trace_path)
            if rank != 0:
                trace_file = f"{base}_rank{rank}{ext or '.csv'}"
        trace = TDTraceLogger(trace_file, rank=rank, enabled=True)
    return trace


def _init_td_output(output_spec: OutputSpec | None, *, rank: int):
    output = None
    output_enabled = False
    out_traj_every = 0
    out_metrics_every = 0
    if output_spec is not None:
        out_traj_every = int(output_spec.traj_every) if output_spec.traj_every else 0
        out_metrics_every = int(output_spec.metrics_every) if output_spec.metrics_every else 0
        output_enabled = (out_traj_every > 0) or (out_metrics_every > 0)
    if output_spec is not None and rank == 0:
        output = make_output_bundle(output_spec)
    return output, output_enabled, out_traj_every, out_metrics_every


def _build_zones_and_automaton(
    *,
    box: float,
    cell_size: float,
    zones_total: int,
    zone_cells_pattern,
    zone_cells_w: int,
    zone_cells_s: int,
    cutoff: float,
    strict_min_zone_width: bool,
    rank: int,
    r: np.ndarray,
    v: np.ndarray,
    comm,
    size: int,
    startup_mode: str,
    traversal: str,
    formal_core: bool,
    debug_invariants: bool,
    max_step_lag: int,
    table_max_age: int,
):
    layout = ZoneLayout1DCells(
        box=box,
        cell_size=cell_size,
        zones_total=zones_total,
        pattern_cells=zone_cells_pattern,
        zone_cells_w=zone_cells_w,
        zone_cells_s=zone_cells_s,
        min_zone_width=float(cutoff),
        strict_min_width=bool(strict_min_zone_width),
    )
    zones_geom = layout.build()
    if rank == 0:
        widths = [float(z.z1 - z.z0) for z in zones_geom if z.n_cells > 0 and z.z1 > z.z0]
        if widths:
            wmin = min(widths)
            wmax = max(widths)
            print(
                f"[info] zone_geom: zones_total={zones_total} width[min,max]=({wmin:.6g},{wmax:.6g}) "
                f"min_zone_width={float(cutoff):.6g} strict={bool(strict_min_zone_width)}",
                flush=True,
            )
    zones = [
        ZoneRuntime(
            zid=z.zid,
            z0=z.z0,
            z1=z.z1,
            ztype=ZoneType.F,
            atom_ids=np.empty((0,), np.int32),
            n_cells=int(getattr(z, "n_cells", 1)),
            step_id=0,
        )
        for z in zones_geom
    ]

    if rank == 0:
        assign_atoms_to_zones(r, zones_geom, box)
        for z in zones:
            z.atom_ids = zones_geom[z.zid].atom_ids
            z.ztype = ZoneType.D if z.atom_ids.size else ZoneType.F

    startup_distribute_zones(comm, rank, size, zones, r, v, startup_mode=startup_mode)
    comm.Barrier()

    bins_cache = PersistentZoneLocalZBinsCache()
    order = traversal_order(zones_total, traversal, rank)
    autom = TDAutomaton1W(
        zones_runtime=zones,
        box=box,
        cutoff=cutoff,
        bins_cache=bins_cache,
        traversal_order=order,
        formal_core=formal_core,
        debug_invariants=debug_invariants,
        max_step_lag=max_step_lag,
        table_max_age=table_max_age,
    )
    return zones, autom


# ------------------ unified wire format (v2.2) ------------------
# payload = int32 nrec, then each record:
#   int32 rectype (0=ZONE_STATE, 1=DELTA, 2=HOLDER_UPDATE, 3=REQ)
#   int32 subtype (for DELTA: 0=MIGRATION, 1=HALO, 2=CORRECTION ...)
#   int32 zid, int32 n, int64 step_id (target zone-time)
#   int32[n] ids, float64[n,3] r, float64[n,3] v
REC_ZONE = 0
REC_DELTA = 1
REC_HOLDER = 2
REC_REQ = 3
REQ_HALO = 0
REQ_HOLDER = 1
DELTA_MIGRATION = 0
DELTA_HALO = 1


def _within_interval_pbc(x: np.ndarray, lo: float, hi: float, box: float) -> np.ndarray:
    lo = float(lo)
    hi = float(hi)
    box = float(box)
    if lo <= hi:
        return (x >= lo) & (x < hi)
    return (x >= lo) | (x < hi)


def halo_filter_by_receiver_aabb(
    r: np.ndarray, ids: np.ndarray, lo: np.ndarray, hi: np.ndarray, pad: float, box: float
) -> np.ndarray:
    pad = float(pad)
    box = float(box)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    lo2 = (lo - pad) % box
    hi2 = (hi + pad) % box
    rr = r[ids]
    m0 = _within_interval_pbc(rr[:, 0], lo2[0], hi2[0], box)
    m1 = _within_interval_pbc(rr[:, 1], lo2[1], hi2[1], box)
    m2 = _within_interval_pbc(rr[:, 2], lo2[2], hi2[2], box)
    return ids[m0 & m1 & m2]


def overlap_filter_by_receiver_aabb(
    r: np.ndarray, ids: np.ndarray, recv_geom, cutoff: float, box: float
) -> np.ndarray:
    return halo_filter_by_receiver_aabb(
        r, ids, recv_geom.lo, recv_geom.hi, float(cutoff), float(box)
    )


def halo_filter_by_receiver_p(
    r: np.ndarray, ids: np.ndarray, z0: float, z1: float, pad: float, box: float
) -> np.ndarray:
    """1D slab halo filter along z with periodic boundaries."""
    z0p = float(z0) - float(pad)
    z1p = float(z1) + float(pad)
    mask = _within_interval_pbc(r[ids, 2], z0p, z1p, float(box))
    return ids[mask]


def pack_records(
    records: list[tuple[int, int, int, np.ndarray, int]], r: np.ndarray, v: np.ndarray
) -> bytes:
    """records: list of (rectype, subtype, zid, ids, step_id)"""
    parts = [np.array([len(records)], dtype=np.int32).tobytes()]
    for rectype, subtype, zid, ids, step_id in records:
        ids = ids.astype(np.int32)
        n = np.int32(ids.size)
        parts.append(
            np.array(
                [np.int32(rectype), np.int32(subtype), np.int32(zid), n], dtype=np.int32
            ).tobytes()
        )
        parts.append(np.array([np.int64(step_id)], dtype=np.int64).tobytes())
        if ids.size:
            parts.append(ids.tobytes())
            parts.append(r[ids].astype(np.float64).tobytes())
            parts.append(v[ids].astype(np.float64).tobytes())
    return b"".join(parts)


def unpack_records(payload: bytes):
    off = 0
    (nrec,) = np.frombuffer(payload[off : off + 4], dtype=np.int32)
    off += 4
    out = []
    for _ in range(int(nrec)):
        rectype, subtype, zid, n = np.frombuffer(payload[off : off + 16], dtype=np.int32)
        off += 16
        (step_id,) = np.frombuffer(payload[off : off + 8], dtype=np.int64)
        off += 8
        rectype = int(rectype)
        subtype = int(subtype)
        zid = int(zid)
        n = int(n)
        step_id = int(step_id)
        if n == 0:
            out.append(
                (
                    rectype,
                    subtype,
                    zid,
                    np.empty((0,), np.int32),
                    np.empty((0, 3)),
                    np.empty((0, 3)),
                    step_id,
                )
            )
            continue
        ids = np.frombuffer(payload[off : off + 4 * n], dtype=np.int32).copy()
        off += 4 * n
        rr = np.frombuffer(payload[off : off + 8 * 3 * n], dtype=np.float64).reshape(n, 3).copy()
        off += 8 * 3 * n
        vv = np.frombuffer(payload[off : off + 8 * 3 * n], dtype=np.float64).reshape(n, 3).copy()
        off += 8 * 3 * n
        out.append((rectype, subtype, zid, ids, rr, vv, step_id))
    return out


def send_payload(comm, dest: int, tag: int, payload: bytes):
    nbytes = np.array([len(payload)], dtype=np.int32)
    comm.Send([nbytes, MPI.INT], dest=dest, tag=tag)
    comm.Send([payload, MPI.BYTE], dest=dest, tag=tag + 1)


def post_send_payload(comm, dest: int, tag: int, payload: bytes):
    """Nonblocking variant; caller must keep returned buffers alive until wait."""
    nbytes = np.array([len(payload)], dtype=np.int32)
    req_n = comm.Isend([nbytes, MPI.INT], dest=dest, tag=tag)
    req_p = comm.Isend([payload, MPI.BYTE], dest=dest, tag=tag + 1)
    return [req_n, req_p], [nbytes, payload]


def recv_payload(comm, source: int, tag: int) -> bytes:
    nbytes = np.empty((1,), dtype=np.int32)
    comm.Recv([nbytes, MPI.INT], source=source, tag=tag)
    buf = bytearray(int(nbytes[0]))
    comm.Recv([buf, MPI.BYTE], source=source, tag=tag + 1)
    return bytes(buf)


def traversal_order(zones_total: int, mode: str, rank: int):
    if mode == "forward":
        return list(range(zones_total))
    if mode == "backward":
        return list(range(zones_total - 1, -1, -1))
    return list(range(zones_total)) if (rank % 2) == 0 else list(range(zones_total - 1, -1, -1))


def startup_distribute_zones(
    comm,
    rank: int,
    size: int,
    zones: list[ZoneRuntime],
    r: np.ndarray,
    v: np.ndarray,
    startup_mode: str,
):
    if startup_mode == "rank0_all":
        if rank == 0:
            for z in zones:
                z.ztype = ZoneType.D if z.atom_ids.size else ZoneType.F
        else:
            for z in zones:
                z.atom_ids = np.empty((0,), np.int32)
                z.ztype = ZoneType.F
        return

    if rank == 0:
        for z in zones:
            owner = z.zid % size
            rec = (REC_ZONE, 0, z.zid, z.atom_ids, z.step_id)
            payload = pack_records([rec], r, v)
            if owner == 0:
                z.ztype = ZoneType.D if z.atom_ids.size else ZoneType.F
            else:
                send_payload(comm, owner, tag=8100 + z.zid, payload=payload)
                z.atom_ids = np.empty((0,), np.int32)
                z.ztype = ZoneType.F
    else:
        for z in zones:
            if (z.zid % size) == rank:
                payload = recv_payload(comm, 0, tag=8100 + z.zid)
                recs = unpack_records(payload)
                rectype, subtype, zid, ids, rr, vv, step_id = recs[0]
                if ids.size:
                    r[ids] = rr
                    v[ids] = vv
                zones[zid].atom_ids = ids
                zones[zid].step_id = step_id
                zones[zid].ztype = ZoneType.D if ids.size else ZoneType.F

    for z in zones:
        if (z.zid % size) != rank and rank != 0:
            z.atom_ids = np.empty((0,), np.int32)
            z.ztype = ZoneType.F


def _write_td_output_step(
    *,
    step: int,
    output_enabled: bool,
    output_due_fn,
    gather_fn,
    output,
    rank: int,
    box: float,
) -> None:
    if not output_enabled:
        return
    due, due_traj, due_metrics = output_due_fn(step)
    if not due:
        return
    r_all, v_all, bmean = gather_fn()
    if rank == 0 and output is not None and r_all is not None and v_all is not None:
        if due_traj and output.traj is not None:
            output.traj.write(step, r_all, v_all, box_value=(float(box), float(box), float(box)))
        if due_metrics and output.metrics is not None:
            output.metrics.write(step, r_all, v_all, buffer_value=bmean, box_value=float(box))


def _run_td_warmup_phase(
    *,
    warmup_steps: int,
    warmup_compute: bool,
    rank: int,
    formal_core: bool,
    batch_size: int,
    overlap_mode: str,
    enable_step_id: bool,
    max_step_lag: int,
    table_max_age: int,
    update_buffers_fn,
    get_skin_global_fn,
    cutoff: float,
    recv_phase_fn,
    can_start_compute_fn,
    start_compute_fn,
    finish_compute_fn,
    send_phase_fn,
    flush_async_sends_fn,
    comm,
    use_verlet: bool,
    verlet_k_steps: int,
    r: np.ndarray,
    v: np.ndarray,
    mass: Union[float, np.ndarray],
    dt: float,
    potential,
) -> None:
    wsteps = max(0, int(warmup_steps))
    if wsteps > 0 and rank == 0:
        print(
            f"[info] warmup: steps={wsteps} compute={warmup_compute} formal_core={formal_core} "
            f"batch_size={batch_size} overlap_mode={overlap_mode} step_id={enable_step_id} "
            f"max_lag={max_step_lag} table_max_age={table_max_age}",
            flush=True,
        )

    for w in range(1, wsteps + 1):
        update_buffers_fn(step=w)
        rc = float(cutoff) + float(get_skin_global_fn())
        if (rank % 2) == 0:
            recv_phase_fn(tag_base=7000 + 10 * w)
            if warmup_compute:
                if can_start_compute_fn():
                    start_compute_fn(
                        r=r,
                        rc=rc,
                        skin_global=(float(get_skin_global_fn()) if use_verlet else 0.0),
                        step=w,
                        verlet_k_steps=(int(verlet_k_steps) if use_verlet else 1),
                    )
                finish_compute_fn(
                    r=r,
                    v=v,
                    mass=mass,
                    dt=dt,
                    potential=potential,
                    cutoff=cutoff,
                    rc=rc,
                    skin_global=(float(get_skin_global_fn()) if use_verlet else 0.0),
                    step=w,
                    verlet_k_steps=(int(verlet_k_steps) if use_verlet else 1),
                    enable_step_id=bool(enable_step_id),
                )
            send_phase_fn(tag_base=7000 + 10 * w, rc=rc, step=w)
        else:
            send_phase_fn(tag_base=7000 + 10 * w, rc=rc, step=w)
            recv_phase_fn(tag_base=7000 + 10 * w)
        flush_async_sends_fn()
        comm.Barrier()


def _run_td_main_phase(
    *,
    n_steps: int,
    rank: int,
    update_buffers_fn,
    get_skin_global_fn,
    cutoff: float,
    recv_phase_fn,
    can_start_compute_fn,
    start_compute_fn,
    finish_compute_fn,
    send_phase_fn,
    apply_ensemble_fn,
    output_enabled: bool,
    output_due_fn,
    gather_fn,
    output,
    box_ref_fn,
    thermo_every: int,
    v: np.ndarray,
    mass: Union[float, np.ndarray],
    r: np.ndarray,
    zones: list[ZoneRuntime],
    autom: TDAutomaton1W,
    batch_size: int,
    use_verlet: bool,
    verlet_k_steps: int,
    dt: float,
    potential,
    cutoff_runtime: float,
    enable_step_id: bool,
) -> None:
    for step in range(1, int(n_steps) + 1):
        s = 1000 + step
        update_buffers_fn(step=s)
        rc = float(cutoff) + float(get_skin_global_fn())

        if (rank % 2) == 0:
            recv_phase_fn(tag_base=6000)

        if can_start_compute_fn():
            start_compute_fn(
                r=r,
                rc=rc,
                skin_global=(float(get_skin_global_fn()) if use_verlet else 0.0),
                step=s,
                verlet_k_steps=(int(verlet_k_steps) if use_verlet else 1),
            )
        finish_compute_fn(
            r=r,
            v=v,
            mass=mass,
            dt=dt,
            potential=potential,
            cutoff=cutoff_runtime,
            rc=rc,
            skin_global=(float(get_skin_global_fn()) if use_verlet else 0.0),
            step=s,
            verlet_k_steps=(int(verlet_k_steps) if use_verlet else 1),
            enable_step_id=bool(enable_step_id),
        )

        send_phase_fn(tag_base=6000, rc=rc, step=s)

        if (rank % 2) != 0:
            recv_phase_fn(tag_base=6000)

        apply_ensemble_fn(step)
        _write_td_output_step(
            step=step,
            output_enabled=output_enabled,
            output_due_fn=output_due_fn,
            gather_fn=gather_fn,
            output=output,
            rank=rank,
            box=float(box_ref_fn()),
        )

        if thermo_every and (step % int(thermo_every) == 0) and rank == 0:
            ke = kinetic_energy(v, mass)
            T = temperature_from_ke(ke, r.shape[0])
            bmean = float(np.mean([zz.buffer for zz in zones]))
            q = len(autom.send_queue)
            try:
                autom.record_wfg_sample()
            except (AttributeError, KeyError, IndexError):
                pass
            diag = autom.diag
            sb = int(diag.get("send_batches", 0))
            sb_total = int(diag.get("send_batch_zones_total", 0))
            sb_max = int(diag.get("send_batch_size_max", 0))
            sb_avg = (float(sb_total) / float(sb)) if sb > 0 else 0.0
            steps_live = [z.step_id for z in zones if z.ztype != ZoneType.F]
            spread = (max(steps_live) - min(steps_live)) if steps_live else 0
            print(
                f"[step={step}] T={T:.3f} skin={float(get_skin_global_fn()):.4f} <b>={bmean:.4f} sendQ={q} rc={rc:.4f} "
                f"mig={diag['migrations']} out={diag.get('outbox_atoms',0)} lagV={diag['viol_lag']} bufV={diag.get('viol_buffer',0)} dA={diag.get('delta_applied',0)} dD={diag.get('delta_deferred',0)} hS={diag.get('halo_sent',0)} hA={diag.get('halo_applied',0)} hD={diag.get('halo_deferred',0)} hG={diag.get('halo_geo_viol',0)} hV={diag.get('halo_support_viol',0)} dX={diag.get('delta_dropped',0)} hX={diag.get('halo_dropped',0)} dO={diag.get('delta_dropped_overflow',0)} "
                f"tblAgeReb={diag['table_rebuild_age']} violW={diag['viol_w_gt1']} violO={diag['viol_send_overlap']} "
                f"zone_step_spread={spread} reqS={diag.get('req_sent',0)} reqR={diag.get('req_rcv',0)} reqHS={diag.get('req_holder_sent',0)} reqHR={diag.get('req_holder_reply',0)} sh={diag.get('shadow_promoted',0)} wT={diag.get('wait_table',0)} wO={diag.get('wait_owner',0)} wOU={diag.get('wait_owner_unknown',0)} wOS={diag.get('wait_owner_stale',0)} "
                f"tbK={batch_size} sb={sb} sbAvg={sb_avg:.2f} sbMax={sb_max} asyncS={diag.get('async_send_msgs',0)} asyncB={diag.get('async_send_bytes',0)} "
                f"sendPackMs={float(diag.get('send_pack_ms',0.0)):.3f} sendWaitMs={float(diag.get('send_wait_ms',0.0)):.3f} recvPollMs={float(diag.get('recv_poll_ms',0.0)):.3f} overlapWinMs={float(diag.get('overlap_window_ms',0.0)):.3f} "
                f"wfgS={diag.get('wfg_samples',0)} wfgC={diag.get('wfg_cycles',0)} wfgO={diag.get('wfg_max_outdeg',0)}",
                flush=True,
            )


def _update_static_3d_halo_geometry_diag(
    ctx: _TDMPICommContext,
    *,
    zid: int,
    halo_ids: np.ndarray,
) -> None:
    if str(ctx.deps_provider_mode) != "static_3d" or halo_ids.size == 0:
        return
    g_z = ctx.geom_aabb_fn(int(ctx.zones[int(zid)].zid))
    m = mask_in_aabb_pbc(
        ctx.r,
        halo_ids.astype(np.int32),
        g_z.lo,
        g_z.hi,
        float(ctx.cutoff),
        float(ctx.box),
    )
    if not bool(m.all()):
        ctx.autom.diag["hG3"] = ctx.autom.diag.get("hG3", 0) + int((~m).sum())


def _cleanup_pending_deltas(ctx: _TDMPICommContext) -> None:
    """Bound pending delta buffer: drop too-old and overflow."""
    to_drop = []
    for (zid, sid, st), parts in ctx.pending_deltas.items():
        z = ctx.zones[zid]
        if z.ztype != ZoneType.F and (int(z.step_id) - int(sid)) > ctx.max_step_lag:
            to_drop.append((zid, sid, st))
    for key in to_drop:
        parts = ctx.pending_deltas.pop(key, [])
        dropped = int(sum(p.size for p in parts))
        ctx.pending_atoms -= dropped
        ctx.autom.diag["delta_dropped"] = ctx.autom.diag.get("delta_dropped", 0) + dropped

    if ctx.max_pending_delta_atoms <= 0:
        return
    while ctx.pending_atoms > ctx.max_pending_delta_atoms and ctx.pending_deltas:
        oldest = min(ctx.pending_deltas.keys(), key=lambda k: k[1])
        parts = ctx.pending_deltas.pop(oldest, [])
        dropped = int(sum(p.size for p in parts))
        ctx.pending_atoms -= dropped
        ctx.autom.diag["delta_dropped_overflow"] = (
            ctx.autom.diag.get("delta_dropped_overflow", 0) + dropped
        )
        ctx.autom.diag["delta_dropped"] = ctx.autom.diag.get("delta_dropped", 0) + dropped


def _apply_pending_if_ready(ctx: _TDMPICommContext, *, zid: int) -> None:
    z = ctx.zones[int(zid)]
    keys = [
        k
        for k in ctx.pending_deltas.keys()
        if int(k[0]) == int(zid) and int(k[1]) == int(z.step_id)
    ]
    if not keys:
        return

    for key in keys:
        _, _, subtype = key
        parts = ctx.pending_deltas.pop(key)
        if not parts:
            continue
        add = (
            np.concatenate(parts).astype(np.int32) if len(parts) > 1 else parts[0].astype(np.int32)
        )
        ctx.pending_atoms -= int(add.size)
        if add.size == 0:
            continue

        if int(subtype) == DELTA_HALO:
            if int(z.halo_step_id) != int(z.step_id):
                z.halo_ids = np.unique(add).astype(np.int32)
                _update_static_3d_halo_geometry_diag(ctx, zid=int(zid), halo_ids=z.halo_ids)
                z.halo_step_id = int(z.step_id)
            else:
                z.halo_ids = np.unique(np.concatenate([z.halo_ids, add]).astype(np.int32))
                _update_static_3d_halo_geometry_diag(ctx, zid=int(zid), halo_ids=z.halo_ids)
            ctx.validate_halo_geometry_fn(int(zid), z.halo_ids)
            ctx.autom.diag["halo_applied"] = ctx.autom.diag.get("halo_applied", 0) + int(add.size)
            continue

        z.atom_ids = (
            add if z.atom_ids.size == 0 else np.concatenate([z.atom_ids, add]).astype(np.int32)
        )
        if z.ztype == ZoneType.F:
            z.ztype = ZoneType.D
        ctx.autom.diag["delta_applied"] = ctx.autom.diag.get("delta_applied", 0) + int(add.size)


def _handle_req_record(
    ctx: _TDMPICommContext,
    *,
    src_rank: int,
    subtype: int,
    dep_zid: int,
    ids: np.ndarray,
) -> None:
    # Request from src_rank:
    #   REQ_HALO   (0): request HALO for dependency dep_zid
    #   REQ_HOLDER (1): request holder-map info for dependency dep_zid
    ctx.autom.diag["req_rcv"] = ctx.autom.diag.get("req_rcv", 0) + 1

    if int(subtype) == REQ_HOLDER:
        ver = int(ctx.holder_ver[dep_zid]) if 0 <= dep_zid < ctx.zones_total else -1
        rec = (REC_HOLDER, 0, dep_zid, np.empty((0,), np.int32), ver)
        ctx.holder_reply_outbox.setdefault(int(src_rank), []).append(rec)
        ctx.autom.diag["req_holder_reply"] = ctx.autom.diag.get("req_holder_reply", 0) + 1
        return

    req_zid = int(ids[0]) if ids.size >= 2 else -1
    if not (
        0 <= dep_zid < ctx.zones_total
        and ctx.zones[dep_zid].ztype != ZoneType.F
        and ctx.zones[dep_zid].atom_ids.size
    ):
        return

    if 0 <= req_zid < ctx.zones_total:
        if str(ctx.deps_provider_mode) == "static_3d":
            g = ctx.geom_aabb_fn(int(req_zid))
            halo_ids = halo_filter_by_receiver_aabb(
                ctx.r,
                ctx.zones[dep_zid].atom_ids.astype(np.int32),
                g.lo,
                g.hi,
                ctx.cutoff,
                ctx.box,
            )
        else:
            halo_ids = halo_filter_by_receiver_p(
                ctx.r,
                ctx.zones[dep_zid].atom_ids.astype(np.int32),
                ctx.zones[req_zid].z0,
                ctx.zones[req_zid].z1,
                ctx.cutoff,
                ctx.box,
            )
    else:
        halo_ids = ctx.zones[dep_zid].atom_ids.astype(np.int32)

    if halo_ids.size:
        rec = (
            REC_DELTA,
            DELTA_HALO,
            dep_zid,
            halo_ids.astype(np.int32),
            int(ctx.zones[dep_zid].step_id),
        )
        ctx.req_reply_outbox.setdefault(int(src_rank), []).append(rec)
        ctx.autom.diag["req_reply_sent"] = ctx.autom.diag.get("req_reply_sent", 0) + 1


def _handle_delta_record(
    ctx: _TDMPICommContext,
    *,
    subtype: int,
    zid: int,
    ids: np.ndarray,
    step_id: int,
    state_before,
) -> None:
    sid = int(step_id)
    z = ctx.zones[int(zid)]
    if int(subtype) == DELTA_HALO and z.ztype == ZoneType.F:
        # Promote to shadow dependency zone carrying only HALO.
        z.ztype = ZoneType.P
        z.step_id = sid
        z.halo_step_id = sid
        z.halo_ids = np.empty((0,), np.int32)
        _update_static_3d_halo_geometry_diag(ctx, zid=int(zid), halo_ids=z.halo_ids)
        ctx.autom.diag["shadow_promoted"] = ctx.autom.diag.get("shadow_promoted", 0) + 1

    if z.ztype != ZoneType.F and int(z.step_id) == sid:
        if ids.size:
            if int(subtype) == DELTA_HALO:
                if int(z.halo_step_id) != int(z.step_id):
                    z.halo_ids = np.unique(ids).astype(np.int32)
                    _update_static_3d_halo_geometry_diag(ctx, zid=int(zid), halo_ids=z.halo_ids)
                    z.halo_step_id = int(z.step_id)
                else:
                    z.halo_ids = np.unique(np.concatenate([z.halo_ids, ids]).astype(np.int32))
                    _update_static_3d_halo_geometry_diag(ctx, zid=int(zid), halo_ids=z.halo_ids)
                ctx.validate_halo_geometry_fn(int(zid), z.halo_ids)
                ctx.autom.diag["halo_applied"] = ctx.autom.diag.get("halo_applied", 0) + int(
                    ids.size
                )
                ctx.trace_event_fn(
                    zid=int(zid),
                    step_id=int(sid),
                    event="RECV_DELTA_HALO",
                    state_before=state_before,
                    state_after=z.ztype,
                    halo_ids_count=int(ids.size),
                    migration_count=0,
                    lag=ctx.lag_value_fn(int(zid)),
                )
            else:
                z.atom_ids = (
                    ids
                    if z.atom_ids.size == 0
                    else np.concatenate([z.atom_ids, ids]).astype(np.int32)
                )
                ctx.autom.diag["delta_applied"] = ctx.autom.diag.get("delta_applied", 0) + int(
                    ids.size
                )
                ctx.trace_event_fn(
                    zid=int(zid),
                    step_id=int(sid),
                    event="RECV_DELTA_MIGRATION",
                    state_before=state_before,
                    state_after=z.ztype,
                    halo_ids_count=0,
                    migration_count=int(ids.size),
                    lag=ctx.lag_value_fn(int(zid)),
                )
        return

    if z.ztype != ZoneType.F and (int(z.step_id) - sid) > ctx.max_step_lag:
        if int(subtype) == DELTA_HALO:
            ctx.autom.diag["halo_dropped"] = ctx.autom.diag.get("halo_dropped", 0) + int(ids.size)
        ctx.autom.diag["delta_dropped"] = ctx.autom.diag.get("delta_dropped", 0) + int(ids.size)
        return

    ctx.pending_deltas.setdefault((int(zid), sid, int(subtype)), []).append(ids.astype(np.int32))
    ctx.pending_atoms += int(ids.size)
    _cleanup_pending_deltas(ctx)
    if int(subtype) == DELTA_HALO:
        ctx.autom.diag["halo_deferred"] = ctx.autom.diag.get("halo_deferred", 0) + int(ids.size)
    else:
        ctx.autom.diag["delta_deferred"] = ctx.autom.diag.get("delta_deferred", 0) + int(ids.size)


def _handle_record(
    ctx: _TDMPICommContext,
    *,
    src_rank: int,
    rectype: int,
    subtype: int,
    zid: int,
    ids: np.ndarray,
    rr: np.ndarray,
    vv: np.ndarray,
    step_id: int,
) -> None:
    state_before = ctx.zones[int(zid)].ztype if 0 <= int(zid) < len(ctx.zones) else None
    if ids.size:
        ctx.r[ids] = rr
        ctx.v[ids] = vv

    if rectype == REC_HOLDER:
        ver = int(step_id)
        if ver > int(ctx.holder_ver[int(zid)]):
            ctx.holder_map[int(zid)] = int(src_rank)
            ctx.holder_ver[int(zid)] = ver
        return

    if rectype == REC_REQ:
        _handle_req_record(
            ctx,
            src_rank=int(src_rank),
            subtype=int(subtype),
            dep_zid=int(zid),
            ids=ids,
        )
        return

    if rectype == REC_ZONE:
        ctx.autom.on_recv(zid, ids, step_id)
        ctx.trace_event_fn(
            zid=int(zid),
            step_id=int(step_id),
            event="RECV_ZONE",
            state_before=state_before,
            state_after=ctx.zones[int(zid)].ztype,
            halo_ids_count=0,
            migration_count=int(ids.size),
            lag=ctx.lag_value_fn(int(zid)),
        )
        _cleanup_pending_deltas(ctx)
        _apply_pending_if_ready(ctx, zid=int(zid))
        return

    _handle_delta_record(
        ctx,
        subtype=int(subtype),
        zid=int(zid),
        ids=ids,
        step_id=int(step_id),
        state_before=state_before,
    )


def _recv_phase(ctx: _TDMPICommContext, *, tag_base: int) -> None:
    _drain_async_sends(ctx, block=True)
    t0 = time.perf_counter()
    # Forward direction: from prev_rank on tag_base.
    while ctx.comm.Iprobe(source=ctx.prev_rank, tag=tag_base):
        payload = recv_payload(ctx.comm, ctx.prev_rank, tag=tag_base)
        for rectype, subtype, zid, ids, rr, vv, step_id in unpack_records(payload):
            _handle_record(
                ctx,
                src_rank=int(ctx.prev_rank),
                rectype=int(rectype),
                subtype=int(subtype),
                zid=int(zid),
                ids=ids,
                rr=rr,
                vv=vv,
                step_id=int(step_id),
            )

    # Backward direction halos: from next_rank on tag_base+2.
    while ctx.comm.Iprobe(source=ctx.next_rank, tag=tag_base + 2):
        payload = recv_payload(ctx.comm, ctx.next_rank, tag=tag_base + 2)
        for rectype, subtype, zid, ids, rr, vv, step_id in unpack_records(payload):
            _handle_record(
                ctx,
                src_rank=int(ctx.next_rank),
                rectype=int(rectype),
                subtype=int(subtype),
                zid=int(zid),
                ids=ids,
                rr=rr,
                vv=vv,
                step_id=int(step_id),
            )
    dt_ms = float((time.perf_counter() - t0) * 1000.0)
    ctx.autom.diag["recv_poll_ms"] = float(ctx.autom.diag.get("recv_poll_ms", 0.0)) + dt_ms


def _drain_async_sends(ctx: _TDMPICommContext, *, block: bool) -> None:
    if not ctx.use_async_send:
        return
    if not ctx.pending_send_reqs:
        ctx.pending_send_post_ts = None
        return

    if ctx.pending_send_post_ts is not None:
        overlap_ms = float((time.perf_counter() - float(ctx.pending_send_post_ts)) * 1000.0)
        if overlap_ms > 0.0:
            ctx.autom.diag["overlap_window_ms"] = float(
                ctx.autom.diag.get("overlap_window_ms", 0.0)
            ) + overlap_ms

    if block:
        t_wait0 = time.perf_counter()
        MPI.Request.Waitall(ctx.pending_send_reqs)
        wait_ms = float((time.perf_counter() - t_wait0) * 1000.0)
        ctx.autom.diag["send_wait_ms"] = float(ctx.autom.diag.get("send_wait_ms", 0.0)) + wait_ms
        ctx.pending_send_reqs.clear()
        ctx.pending_send_keepalive.clear()
        ctx.pending_send_post_ts = None
        return

    all_done, _ = MPI.Request.Testall(ctx.pending_send_reqs)
    if bool(all_done):
        ctx.pending_send_reqs.clear()
        ctx.pending_send_keepalive.clear()
        ctx.pending_send_post_ts = None


def _send_phase(ctx: _TDMPICommContext, *, tag_base: int, rc: float, step: int) -> None:
    # Build per-destination record buckets for direct routing.
    buckets: dict[int, list[_WireRecord]] = {}

    if ctx.req_reply_outbox:
        for dest_rank, recs in list(ctx.req_reply_outbox.items()):
            if recs:
                buckets.setdefault(int(dest_rank), []).extend(recs)
        ctx.req_reply_outbox.clear()
    if ctx.holder_reply_outbox:
        for dest_rank, recs in list(ctx.holder_reply_outbox.items()):
            if recs:
                buckets.setdefault(int(dest_rank), []).extend(recs)
        ctx.holder_reply_outbox.clear()

    def add_record(dest_rank: int, rec: _WireRecord) -> None:
        buckets.setdefault(int(dest_rank), []).append(rec)

    batch = ctx.autom.pop_send_batch(batch_size=ctx.batch_size)
    while batch:
        for zid in batch:
            z = ctx.zones[int(zid)]
            if z.ztype != ZoneType.S:
                continue
            z_state_before = z.ztype

            deps = ctx.deps_zone_ids_fn(int(zid))
            for dep_zid in deps:
                try:
                    overlap_ids = ctx.overlap_set_for_send_fn(
                        int(zid),
                        int(dep_zid),
                        float(rc),
                        int(step),
                    )
                except (KeyError, IndexError, ValueError):
                    overlap_ids = set()
                ov_ids = (
                    np.array(list(overlap_ids), dtype=np.int32)
                    if overlap_ids
                    else z.atom_ids.astype(np.int32)
                )
                if str(ctx.deps_provider_mode) == "static_3d":
                    g = ctx.geom_aabb_fn(int(dep_zid))
                    halo_ids = halo_filter_by_receiver_aabb(
                        ctx.r,
                        ov_ids,
                        g.lo,
                        g.hi,
                        ctx.cutoff,
                        ctx.box,
                    )
                else:
                    halo_ids = halo_filter_by_receiver_p(
                        ctx.r,
                        ov_ids,
                        ctx.zones[int(dep_zid)].z0,
                        ctx.zones[int(dep_zid)].z1,
                        ctx.cutoff,
                        ctx.box,
                    )
                if halo_ids.size:
                    dest = ctx.holder_map[int(dep_zid)]
                    if dest < 0:
                        dest = ctx.next_rank
                    add_record(
                        int(dest),
                        (REC_DELTA, DELTA_HALO, int(dep_zid), halo_ids, int(z.step_id)),
                    )
                    ctx.autom.diag["halo_sent"] = ctx.autom.diag.get("halo_sent", 0) + int(
                        halo_ids.size
                    )
                    ctx.trace_event_fn(
                        zid=int(zid),
                        step_id=int(z.step_id),
                        event="SEND_DELTA_HALO",
                        state_before=z_state_before,
                        state_after=z_state_before,
                        halo_ids_count=int(halo_ids.size),
                        migration_count=0,
                        lag=ctx.lag_value_fn(int(zid)),
                    )

            if ctx.static_rr:
                z.ztype = ZoneType.D if z.atom_ids.size else ZoneType.F
                ctx.trace_event_fn(
                    zid=int(zid),
                    step_id=int(z.step_id),
                    event="SEND_ZONE",
                    state_before=z_state_before,
                    state_after=z.ztype,
                    halo_ids_count=0,
                    migration_count=0,
                    lag=ctx.lag_value_fn(int(zid)),
                )
            else:
                next_zid = (int(zid) + 1) % ctx.zones_total
                overlap_next = ctx.overlap_set_for_send_fn(
                    int(zid), int(next_zid), float(rc), int(step)
                )
                if overlap_next:
                    mask_send = np.array(
                        [int(a) not in overlap_next for a in z.atom_ids], dtype=bool
                    )
                    send_ids = z.atom_ids[mask_send].astype(np.int32)
                    keep_ids = z.atom_ids[~mask_send].astype(np.int32)
                else:
                    send_ids = z.atom_ids.astype(np.int32)
                    keep_ids = np.empty((0,), np.int32)

                if ctx.debug_invariants and overlap_next:
                    if any(int(a) in overlap_next for a in send_ids.tolist()):
                        ctx.autom.diag["viol_send_overlap"] += 1

                add_record(int(ctx.next_rank), (REC_ZONE, 0, int(zid), send_ids, int(z.step_id)))
                ctx.set_holder_fn(int(zid), int(ctx.next_rank))

                z.atom_ids = keep_ids
                z.table = None
                z.ztype = ZoneType.D if keep_ids.size else ZoneType.F
                ctx.trace_event_fn(
                    zid=int(zid),
                    step_id=int(z.step_id),
                    event="SEND_ZONE",
                    state_before=z_state_before,
                    state_after=z.ztype,
                    halo_ids_count=0,
                    migration_count=int(send_ids.size),
                    lag=ctx.lag_value_fn(int(zid)),
                )

        batch = ctx.autom.pop_send_batch(batch_size=ctx.batch_size)

    for dest_zid, sid, ids in ctx.autom.iter_outbox_records():
        if ids.size:
            dest = ctx.holder_map[int(dest_zid)]
            if dest < 0:
                dest = ctx.next_rank
            add_record(int(dest), (REC_DELTA, DELTA_MIGRATION, int(dest_zid), ids, int(sid)))
            ctx.trace_event_fn(
                zid=int(dest_zid),
                step_id=int(sid),
                event="SEND_DELTA_MIGRATION",
                state_before=(
                    ctx.zones[int(dest_zid)].ztype if 0 <= int(dest_zid) < ctx.zones_total else None
                ),
                state_after=(
                    ctx.zones[int(dest_zid)].ztype if 0 <= int(dest_zid) < ctx.zones_total else None
                ),
                halo_ids_count=0,
                migration_count=int(ids.size),
                lag=ctx.lag_value_fn(int(dest_zid)) if 0 <= int(dest_zid) < ctx.zones_total else 0,
            )
    ctx.autom.clear_outbox()

    t_pack_ms = 0.0
    for dest_rank, recs in buckets.items():
        if not recs:
            continue
        t_pack0 = time.perf_counter()
        payload = pack_records(recs, ctx.r, ctx.v)
        t_pack_ms += float((time.perf_counter() - t_pack0) * 1000.0)
        if ctx.use_async_send:
            reqs, refs = post_send_payload(ctx.comm, int(dest_rank), tag=tag_base, payload=payload)
            if reqs:
                if ctx.pending_send_post_ts is None:
                    ctx.pending_send_post_ts = time.perf_counter()
                ctx.pending_send_reqs.extend(reqs)
            if refs:
                ctx.pending_send_keepalive.extend(refs)
            ctx.autom.diag["async_send_msgs"] = ctx.autom.diag.get("async_send_msgs", 0) + 1
            ctx.autom.diag["async_send_bytes"] = ctx.autom.diag.get("async_send_bytes", 0) + int(
                len(payload)
            )
        else:
            send_payload(ctx.comm, int(dest_rank), tag=tag_base, payload=payload)
    ctx.autom.diag["send_pack_ms"] = float(ctx.autom.diag.get("send_pack_ms", 0.0)) + float(
        t_pack_ms
    )


@dataclass
class _TDMPISimState:
    """Mutable simulation state shared across helper functions."""

    box: float
    skin_global: float = 0.0


# ---------------------------------------------------------------------------
# Helpers extracted from _run_td_full_mpi_1d_legacy closures
# ---------------------------------------------------------------------------


def _trace_event_impl(
    trace: TDTraceLogger | None,
    autom: TDAutomaton1W,
    *,
    zid: int,
    step_id: int,
    event: str,
    state_before=None,
    state_after=None,
    halo_ids_count: int = 0,
    migration_count: int = 0,
    lag: int = 0,
) -> None:
    if trace is None:
        return
    trace.log(
        step_id=int(step_id),
        zone_id=int(zid),
        event=str(event),
        state_before=state_before,
        state_after=state_after,
        halo_ids_count=int(halo_ids_count),
        migration_count=int(migration_count),
        lag=int(lag),
        invariant_flags=format_invariant_flags(autom.diag),
    )


def _lag_value_impl(
    autom: TDAutomaton1W,
    zones: list[ZoneRuntime],
    zid: int,
) -> int:
    try:
        deps, _, _ = autom._deps(int(zid))
    except (KeyError, IndexError, ValueError):
        deps = []
    if autom.deps_table_func is not None:
        try:
            deps = list(autom.deps_table_func(int(zid)))
        except (KeyError, IndexError, ValueError):
            deps = list(deps)
    zt = zones[int(zid)].step_id
    lags = []
    for did in deps:
        dz = zones[int(did)]
        if dz.ztype != ZoneType.F:
            lags.append(int(zt - dz.step_id))
    return max(lags) if lags else 0


def _start_compute_with_trace_impl(
    trace: TDTraceLogger | None,
    autom: TDAutomaton1W,
    zones: list[ZoneRuntime],
    *,
    r: np.ndarray,
    rc: float,
    skin_global: float,
    step: int,
    verlet_k_steps: int,
):
    pre = [z.ztype for z in zones] if trace is not None else None
    zid = autom.start_compute(
        r=r, rc=rc, skin_global=skin_global, step=step, verlet_k_steps=verlet_k_steps
    )
    if zid is not None and trace is not None and pre is not None:
        _trace_event_impl(
            trace,
            autom,
            zid=zid,
            step_id=int(zones[zid].step_id),
            event="START_COMPUTE",
            state_before=pre[zid],
            state_after=zones[zid].ztype,
            halo_ids_count=int(zones[zid].halo_ids.size),
            migration_count=0,
            lag=_lag_value_impl(autom, zones, zid),
        )
        for did in autom._locked_donors.get(zid, []):
            _trace_event_impl(
                trace,
                autom,
                zid=did,
                step_id=int(zones[did].step_id),
                event="LOCK_DONOR",
                state_before=pre[did],
                state_after=zones[did].ztype,
                halo_ids_count=int(zones[did].halo_ids.size),
                migration_count=0,
                lag=_lag_value_impl(autom, zones, did),
            )
    return zid


def _finish_compute_with_trace_impl(
    trace: TDTraceLogger | None,
    autom: TDAutomaton1W,
    zones: list[ZoneRuntime],
    atom_types,
    *,
    r: np.ndarray,
    v: np.ndarray,
    mass: Union[float, np.ndarray],
    dt: float,
    potential,
    cutoff: float,
    rc: float,
    skin_global: float,
    step: int,
    verlet_k_steps: int,
    enable_step_id: bool = True,
):
    pre = [z.ztype for z in zones] if trace is not None else None
    zid = autom.compute_step_for_work_zone(
        r=r,
        v=v,
        mass=mass,
        dt=dt,
        potential=potential,
        cutoff=cutoff,
        rc=rc,
        skin_global=skin_global,
        step=step,
        verlet_k_steps=verlet_k_steps,
        atom_types=atom_types,
        enable_step_id=enable_step_id,
    )
    if zid is not None and trace is not None and pre is not None:
        _trace_event_impl(
            trace,
            autom,
            zid=zid,
            step_id=int(zones[zid].step_id),
            event="FINISH_COMPUTE",
            state_before=pre[zid],
            state_after=zones[zid].ztype,
            halo_ids_count=0,
            migration_count=int(autom.diag.get("migrations", 0)),
            lag=_lag_value_impl(autom, zones, zid),
        )
        for did, pstate in enumerate(pre):
            if pstate == ZoneType.P and zones[did].ztype == ZoneType.D:
                _trace_event_impl(
                    trace,
                    autom,
                    zid=did,
                    step_id=int(zones[did].step_id),
                    event="RELEASE_DONOR",
                    state_before=pstate,
                    state_after=zones[did].ztype,
                    halo_ids_count=int(zones[did].halo_ids.size),
                    migration_count=0,
                    lag=_lag_value_impl(autom, zones, did),
                )
    return zid


def _build_deps_provider_3d(
    deps_mode: str,
    autom: TDAutomaton1W,
    *,
    zones_nx: int,
    zones_ny: int,
    zones_nz: int,
    box: float,
    cutoff: float,
    mpi_size: int,
):
    if deps_mode != "static_3d":
        return None

    deps_provider_3d = DepsProvider3DBlock(
        nx=int(zones_nx),
        ny=int(zones_ny),
        nz=int(zones_nz),
        box=(float(box), float(box), float(box)),
        cutoff=float(cutoff),
        mpi_size=int(mpi_size),
    )

    class _GeomProvider:
        def __init__(self, base):
            self.base = base

        def geom(self, zid: int) -> ZoneGeomAABB:
            lo, hi = self.base.geom(int(zid))
            return ZoneGeomAABB(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))

    autom.geom_provider = _GeomProvider(deps_provider_3d)
    return deps_provider_3d


def _geom_aabb_impl(deps_provider_3d, zid: int) -> ZoneGeomAABB:
    if deps_provider_3d is None:
        raise RuntimeError("static_3d required for AABB geometry")
    lo, hi = deps_provider_3d.geom(int(zid))
    return ZoneGeomAABB(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))


def _deps_zone_ids_impl(
    deps_provider_3d,
    zones: list[ZoneRuntime],
    cutoff: float,
    sim: _TDMPISimState,
    zid: int,
) -> list[int]:
    if deps_provider_3d is not None:
        return [int(d) for d in deps_provider_3d.deps_table(int(zid))]
    z = zones[int(zid)]
    deps = zones_overlapping_range_pbc(z.z0 - cutoff, z.z1 + cutoff, sim.box, zones)
    return [int(d) for d in deps if int(d) != int(zid)]


def _owner_deps_zone_ids_impl(
    deps_provider_3d,
    zones: list[ZoneRuntime],
    cutoff: float,
    owner_buffer: float,
    sim: _TDMPISimState,
    zid: int,
) -> list[int]:
    if deps_provider_3d is not None:
        return [int(d) for d in deps_provider_3d.deps_owner(int(zid))]
    z = zones[int(zid)]
    buf = float(owner_buffer) if owner_buffer and owner_buffer > 0 else float(cutoff)
    if owner_buffer <= 0:
        buf = max(buf, float(getattr(z, "buffer", 0.0)))
    deps = zones_overlapping_range_pbc(z.z0 - buf, z.z1 + buf, sim.box, zones)
    return [int(d) for d in deps if int(d) != int(zid)]


def _init_holder_map(
    static_rr: bool,
    zones_total: int,
    zones: list[ZoneRuntime],
    comm,
    rank: int,
    size: int,
) -> tuple[list[int], list[int], int]:
    if static_rr:
        holder_map = [int(zid) % int(size) for zid in range(zones_total)]
        holder_ver = [0 for _ in range(zones_total)]
        return holder_map, holder_ver, 0
    local_have = [z.zid for z in zones if z.ztype != ZoneType.F]
    all_have = comm.allgather(local_have)
    holder_map = [-1 for _ in range(zones_total)]
    holder_ver = [-1 for _ in range(zones_total)]
    for rrk, zlist in enumerate(all_have):
        for zid in zlist:
            holder_map[int(zid)] = int(rrk)
            holder_ver[int(zid)] = 0
    return holder_map, holder_ver, 0


def _update_buffers_impl(
    zones: list[ZoneRuntime],
    v: np.ndarray,
    dt: float,
    buffer_k: float,
    skin_from_buffer: bool,
    max_step_lag: int,
    autom: TDAutomaton1W,
    sim: _TDMPISimState,
    step: int,
) -> None:
    sim.skin_global = 0.0
    for z in zones:
        if z.ztype == ZoneType.F or z.atom_ids.size == 0:
            z.buffer = 0.0
            z.skin = 0.0
            continue
        b, sk = compute_zone_buffer_skin(
            v,
            z.atom_ids,
            dt,
            buffer_k,
            skin_from_buffer=skin_from_buffer,
            lag_steps=max_step_lag,
        )
        z.buffer = b
        z.skin = sk
        if max_step_lag > 0:
            speeds = np.linalg.norm(v[z.atom_ids], axis=1)
            vmax = float(speeds.max()) if speeds.size else 0.0
            required = vmax * dt * float(max_step_lag + 1)
            if b + FLOAT_EQ_ATOL < required:
                autom.diag["viol_buffer"] = autom.diag.get("viol_buffer", 0) + 1
        sim.skin_global = max(sim.skin_global, sk)


def _scale_zone_geometry_impl(
    zones: list[ZoneRuntime],
    autom: TDAutomaton1W,
    cutoff: float,
    lam: float,
    new_box: float,
) -> None:
    if abs(float(lam) - 1.0) <= FLOAT_EQ_ATOL:
        autom.box = float(new_box)
        autom.zwidth = autom.box / max(1, len(zones))
        return
    for z in zones:
        z.z0 = float(z.z0) * float(lam)
        z.z1 = float(z.z1) * float(lam)
    widths = [float(z.z1 - z.z0) for z in zones if z.n_cells > 0 and z.z1 > z.z0]
    if widths and (min(widths) + GEOM_EPSILON < float(cutoff)):
        raise ValueError("NPT scaling violated zone width >= cutoff in td_full_mpi layout")
    autom.box = float(new_box)
    autom.zwidth = autom.box / max(1, len(zones))


def _apply_ensemble_impl(
    ensemble,
    zones: list[ZoneRuntime],
    autom: TDAutomaton1W,
    sim: _TDMPISimState,
    *,
    r: np.ndarray,
    v: np.ndarray,
    mass: Union[float, np.ndarray],
    potential,
    cutoff: float,
    atom_types,
    dt: float,
    step: int,
) -> None:
    new_box, _lam_t, lam_b = apply_ensemble_step(
        step=int(step),
        ensemble=ensemble,
        r=r,
        v=v,
        mass=mass,
        box=float(sim.box),
        potential=potential,
        cutoff=cutoff,
        atom_types=atom_types,
        dt=dt,
    )
    if ensemble.kind == "npt":
        _scale_zone_geometry_impl(zones, autom, cutoff, lam_b, new_box)
        sim.box = float(new_box)


def _output_due_impl(
    output_enabled: bool,
    out_traj_every: int,
    out_metrics_every: int,
    step: int,
):
    if not output_enabled:
        return False, False, False
    due_traj = (out_traj_every > 0) and (step % out_traj_every == 0)
    due_metrics = (out_metrics_every > 0) and (step % out_metrics_every == 0)
    return (due_traj or due_metrics), due_traj, due_metrics


def _gather_for_output_impl(
    zones: list[ZoneRuntime],
    comm,
    r: np.ndarray,
    v: np.ndarray,
    rank: int,
):
    owned = [z.atom_ids for z in zones if z.ztype != ZoneType.F and z.atom_ids.size]
    owned_ids = np.concatenate(owned).astype(np.int32) if owned else np.empty((0,), np.int32)
    ids_list = comm.gather(owned_ids, root=0)
    r_list = comm.gather(r[owned_ids] if owned_ids.size else np.empty((0, 3), float), root=0)
    v_list = comm.gather(v[owned_ids] if owned_ids.size else np.empty((0, 3), float), root=0)
    bsum = float(sum(z.buffer for z in zones if z.atom_ids.size))
    bcount = int(sum(1 for z in zones if z.atom_ids.size))
    bsum_list = comm.gather(bsum, root=0)
    bcount_list = comm.gather(bcount, root=0)
    if rank != 0:
        return None, None, 0.0
    r_all = np.zeros_like(r)
    v_all = np.zeros_like(v)
    if ids_list is not None:
        for ids, rr, vv in zip(ids_list, r_list, v_list):
            if ids is None or len(ids) == 0:
                continue
            r_all[ids] = rr
            v_all[ids] = vv
    total_bsum = float(sum(bsum_list)) if bsum_list is not None else 0.0
    total_bcount = int(sum(bcount_list)) if bcount_list is not None else 0
    bmean = total_bsum / max(1, total_bcount)
    return r_all, v_all, bmean


def _validate_halo_geometry_impl(
    zones: list[ZoneRuntime],
    r: np.ndarray,
    cutoff: float,
    sim: _TDMPISimState,
    autom: TDAutomaton1W,
    zid: int,
    halo_ids: np.ndarray,
) -> None:
    if halo_ids.size == 0:
        return
    z = zones[int(zid)]
    z0p = z.z0 - cutoff
    z1p = z.z1 + cutoff
    mask = _within_interval_pbc(r[halo_ids, 2], z0p, z1p, sim.box)
    bad = int((~mask).sum())
    if bad > 0:
        autom.diag["halo_geo_viol"] = autom.diag.get("halo_geo_viol", 0) + bad
    if getattr(z, "table", None) is not None and z.table is not None:
        supp = set(map(int, z.table.support_ids().tolist()))
        viol = sum(1 for atom_id in halo_ids.tolist() if int(atom_id) not in supp)
        if viol:
            autom.diag["halo_support_viol"] = autom.diag.get("halo_support_viol", 0) + int(viol)


def _overlap_set_for_send_impl(
    zones: list[ZoneRuntime],
    r: np.ndarray,
    overlap_mode: str,
    deps_provider_mode: str,
    deps_provider_3d,
    sim: _TDMPISimState,
    use_verlet: bool,
    verlet_k_steps: int,
    autom: TDAutomaton1W,
    zid: int,
    next_zid: int,
    rc: float,
    step: int,
) -> set:
    if zones[zid].atom_ids.size == 0:
        return set()
    if overlap_mode == "geometric_rc":
        if str(deps_provider_mode) == "static_3d":
            gR = _geom_aabb_impl(deps_provider_3d, int(next_zid))
            ov = overlap_filter_by_receiver_aabb(
                r, zones[zid].atom_ids.astype(np.int32), gR, rc, sim.box
            )
            return set(map(int, ov.tolist()))
        ov = halo_filter_by_receiver_p(
            r,
            zones[zid].atom_ids.astype(np.int32),
            zones[next_zid].z0,
            zones[next_zid].z1,
            rc,
            sim.box,
        )
        return set(map(int, ov.tolist()))

    autom.ensure_table(
        next_zid,
        r=r,
        rc=rc,
        skin_global=(sim.skin_global if use_verlet else 0.0),
        step=step,
        verlet_k_steps=(verlet_k_steps if use_verlet else 1),
    )
    support = autom.table_support(next_zid)
    if support.size == 0:
        return set()
    return set(map(int, np.intersect1d(zones[zid].atom_ids, support, assume_unique=False).tolist()))


# ---------------------------------------------------------------------------
# Orchestrator: _run_td_full_mpi_1d_legacy (thin wiring only)
# ---------------------------------------------------------------------------


def _run_td_full_mpi_1d_legacy(
    r: np.ndarray,
    v: np.ndarray,
    mass: Union[float, np.ndarray],
    box: float,
    potential,
    dt: float,
    cutoff: float,
    n_steps: int,
    *,
    config: TDFullMPIRunConfig,
    output_spec: OutputSpec | None = None,
):
    # --- MPI / backend / ensemble bootstrap ---
    runtime = _init_td_full_mpi_runtime(
        device=str(config.device),
        ensemble_kind=str(config.ensemble_kind),
        thermostat=config.thermostat,
        barostat=config.barostat,
        cuda_aware_mpi=bool(config.cuda_aware_mpi),
        comm_overlap_isend=bool(config.comm_overlap_isend),
        batch_size=int(config.batch_size),
    )
    batch_size = int(runtime.batch_size)
    comm = runtime.comm
    rank = int(runtime.rank)
    size = int(runtime.size)
    prev_rank = int(runtime.prev_rank)
    next_rank = int(runtime.next_rank)
    ensemble = runtime.ensemble
    use_async_send = bool(runtime.use_async_send)

    atom_types = normalize_atom_types(config.atom_types, n_atoms=r.shape[0])
    potential_compute = _wrap_potential_for_gpu_refinement(
        potential=potential, backend=runtime.backend
    )

    trace = _init_td_trace(
        trace_enabled=bool(config.trace_enabled),
        trace_path=str(config.trace_path),
        rank=rank,
        size=size,
    )
    output, output_enabled, out_traj_every, out_metrics_every = _init_td_output(
        output_spec,
        rank=rank,
    )

    # --- validation ---
    deps_mode = str(config.deps_provider_mode)
    static_rr = deps_mode == "static_rr"
    if static_rr and config.startup_mode != "scatter_zones":
        raise RuntimeError("static_rr requires startup_mode=scatter_zones (static ownership)")
    if config.strict_fast_sync:
        if not (
            config.fast_sync
            and config.zones_total == 2 * size
            and config.startup_mode == "scatter_zones"
        ):
            raise RuntimeError(
                "strict_fast_sync requires fast_sync=true, zones_total=2P, startup_mode=scatter_zones"
            )

    # --- zones + automaton ---
    zones, autom = _build_zones_and_automaton(
        box=float(box),
        cell_size=float(config.cell_size),
        zones_total=int(config.zones_total),
        zone_cells_pattern=config.zone_cells_pattern,
        zone_cells_w=int(config.zone_cells_w),
        zone_cells_s=int(config.zone_cells_s),
        cutoff=float(cutoff),
        strict_min_zone_width=bool(config.strict_min_zone_width),
        rank=rank,
        r=r,
        v=v,
        comm=comm,
        size=size,
        startup_mode=str(config.startup_mode),
        traversal=str(config.traversal),
        formal_core=bool(config.formal_core),
        debug_invariants=bool(config.debug_invariants),
        max_step_lag=int(config.max_step_lag),
        table_max_age=int(config.table_max_age),
    )

    # --- mutable simulation state (replaces nonlocal box / skin_global) ---
    sim = _TDMPISimState(box=float(box))

    # --- 3D deps provider ---
    deps_provider_3d = _build_deps_provider_3d(
        deps_mode,
        autom,
        zones_nx=int(config.zones_nx),
        zones_ny=int(config.zones_ny),
        zones_nz=int(config.zones_nz),
        box=float(box),
        cutoff=float(cutoff),
        mpi_size=size,
    )

    # --- deps wiring ---
    require_table_deps = bool(config.require_table_deps or config.require_local_deps)
    autom.set_deps_funcs(
        table_func=lambda zid: _deps_zone_ids_impl(deps_provider_3d, zones, cutoff, sim, int(zid)),
        owner_func=lambda zid: _owner_deps_zone_ids_impl(
            deps_provider_3d, zones, cutoff, config.owner_buffer, sim, int(zid)
        ),
    )

    # --- holder map ---
    holder_map, holder_ver, holder_epoch = _init_holder_map(
        static_rr,
        int(config.zones_total),
        zones,
        comm,
        rank,
        size,
    )

    # --- deps predicates ---
    if static_rr and config.require_owner_deps:
        owner_pred = lambda did: holder_map[int(did)] != -1
    else:
        owner_pred = lambda did: (
            (
                (holder_map[int(did)] != -1)
                and (
                    (not config.require_owner_ver)
                    or (holder_ver[int(did)] >= (holder_epoch - (2 * config.max_step_lag + 2)))
                )
            )
            if config.require_owner_deps
            else True
        )
    autom.set_deps_preds(
        table_pred=(
            lambda did: (zones[int(did)].ztype != ZoneType.F) if require_table_deps else True
        ),
        owner_pred=owner_pred,
    )

    # --- outbox / pending state ---
    req_reply_outbox: dict[int, list[tuple[int, int, int, np.ndarray, int]]] = {}
    holder_reply_outbox: dict[int, list[tuple[int, int, int, np.ndarray, int]]] = {}
    pending_deltas: dict[tuple[int, int, int], list[np.ndarray]] = {}

    # --- thin closure adapters (bind top-level helpers to local state) ---
    def _trace_event(**kw):
        _trace_event_impl(trace, autom, **kw)

    def _lag_value(zid: int) -> int:
        return _lag_value_impl(autom, zones, zid)

    def _start_compute_with_trace(**kw):
        return _start_compute_with_trace_impl(trace, autom, zones, **kw)

    def _finish_compute_with_trace(**kw):
        return _finish_compute_with_trace_impl(trace, autom, zones, atom_types, **kw)

    def _geom_aabb(zid: int) -> ZoneGeomAABB:
        return _geom_aabb_impl(deps_provider_3d, zid)

    def deps_zone_ids(zid: int) -> list[int]:
        return _deps_zone_ids_impl(deps_provider_3d, zones, cutoff, sim, zid)

    def _set_holder(zid: int, rk: int):
        if static_rr:
            return
        holder_map[int(zid)] = int(rk)

    def update_buffers(step: int):
        _update_buffers_impl(
            zones,
            v,
            dt,
            config.buffer_k,
            config.skin_from_buffer,
            config.max_step_lag,
            autom,
            sim,
            step,
        )

    def _apply_ensemble(step: int):
        _apply_ensemble_impl(
            ensemble,
            zones,
            autom,
            sim,
            r=r,
            v=v,
            mass=mass,
            potential=potential,
            cutoff=cutoff,
            atom_types=atom_types,
            dt=dt,
            step=step,
        )

    def _output_due(step: int):
        return _output_due_impl(output_enabled, out_traj_every, out_metrics_every, step)

    def _gather_for_output():
        return _gather_for_output_impl(zones, comm, r, v, rank)

    def _validate_halo_geometry(zid: int, halo_ids: np.ndarray):
        _validate_halo_geometry_impl(zones, r, cutoff, sim, autom, zid, halo_ids)

    def overlap_set_for_send(zid: int, next_zid: int, rc: float, step: int) -> set:
        return _overlap_set_for_send_impl(
            zones,
            r,
            config.overlap_mode,
            config.deps_provider_mode,
            deps_provider_3d,
            sim,
            config.use_verlet,
            config.verlet_k_steps,
            autom,
            zid,
            next_zid,
            rc,
            step,
        )

    # --- comm context ---
    comm_ctx = _TDMPICommContext(
        comm=comm,
        prev_rank=prev_rank,
        next_rank=next_rank,
        use_async_send=bool(use_async_send),
        batch_size=int(batch_size),
        box=float(sim.box),
        cutoff=float(cutoff),
        deps_provider_mode=str(config.deps_provider_mode),
        static_rr=bool(static_rr),
        debug_invariants=bool(config.debug_invariants),
        max_step_lag=int(config.max_step_lag),
        max_pending_delta_atoms=int(config.max_pending_delta_atoms),
        r=r,
        v=v,
        zones=zones,
        zones_total=int(config.zones_total),
        autom=autom,
        holder_map=holder_map,
        holder_ver=holder_ver,
        req_reply_outbox=req_reply_outbox,
        holder_reply_outbox=holder_reply_outbox,
        pending_deltas=pending_deltas,
        pending_atoms=0,
        geom_aabb_fn=_geom_aabb,
        deps_zone_ids_fn=deps_zone_ids,
        overlap_set_for_send_fn=overlap_set_for_send,
        trace_event_fn=_trace_event,
        lag_value_fn=_lag_value,
        set_holder_fn=_set_holder,
        validate_halo_geometry_fn=_validate_halo_geometry,
        pending_send_reqs=[],
        pending_send_keepalive=[],
        pending_send_post_ts=None,
    )

    def recv_phase(tag_base: int) -> None:
        # Keep communication geometry aligned with the current box (NPT can rescale it).
        comm_ctx.box = float(sim.box)
        _recv_phase(comm_ctx, tag_base=int(tag_base))

    def send_phase(tag_base: int, rc: float, step: int) -> None:
        # Keep communication geometry aligned with the current box (NPT can rescale it).
        comm_ctx.box = float(sim.box)
        _send_phase(comm_ctx, tag_base=int(tag_base), rc=float(rc), step=int(step))

    def flush_async_sends() -> None:
        _drain_async_sends(comm_ctx, block=True)

    try:
        # --- initial output ---
        if output_enabled:
            update_buffers(step=0)
            _write_td_output_step(
                step=0,
                output_enabled=output_enabled,
                output_due_fn=_output_due,
                gather_fn=_gather_for_output,
                output=output,
                rank=rank,
                box=float(sim.box),
            )

        # --- warmup ---
        _run_td_warmup_phase(
            warmup_steps=config.warmup_steps,
            warmup_compute=bool(config.warmup_compute),
            rank=rank,
            formal_core=bool(config.formal_core),
            batch_size=int(batch_size),
            overlap_mode=str(config.overlap_mode),
            enable_step_id=bool(config.enable_step_id),
            max_step_lag=int(config.max_step_lag),
            table_max_age=int(config.table_max_age),
            update_buffers_fn=update_buffers,
            get_skin_global_fn=lambda: sim.skin_global,
            cutoff=float(cutoff),
            recv_phase_fn=recv_phase,
            can_start_compute_fn=autom.can_start_compute,
            start_compute_fn=_start_compute_with_trace,
            finish_compute_fn=_finish_compute_with_trace,
            send_phase_fn=send_phase,
            flush_async_sends_fn=flush_async_sends,
            comm=comm,
            use_verlet=bool(config.use_verlet),
            verlet_k_steps=int(config.verlet_k_steps),
            r=r,
            v=v,
            mass=mass,
            dt=float(dt),
            potential=potential_compute,
        )

        # --- main simulation ---
        _run_td_main_phase(
            n_steps=int(n_steps),
            rank=rank,
            update_buffers_fn=update_buffers,
            get_skin_global_fn=lambda: sim.skin_global,
            cutoff=float(cutoff),
            recv_phase_fn=recv_phase,
            can_start_compute_fn=autom.can_start_compute,
            start_compute_fn=_start_compute_with_trace,
            finish_compute_fn=_finish_compute_with_trace,
            send_phase_fn=send_phase,
            apply_ensemble_fn=_apply_ensemble,
            output_enabled=bool(output_enabled),
            output_due_fn=_output_due,
            gather_fn=_gather_for_output,
            output=output,
            box_ref_fn=lambda: sim.box,
            thermo_every=int(config.thermo_every),
            v=v,
            mass=mass,
            r=r,
            zones=zones,
            autom=autom,
            batch_size=int(batch_size),
            use_verlet=bool(config.use_verlet),
            verlet_k_steps=int(config.verlet_k_steps),
            dt=float(dt),
            potential=potential_compute,
            cutoff_runtime=float(cutoff),
            enable_step_id=bool(config.enable_step_id),
        )
    finally:
        # Ensure MPI send requests are drained even on failures.
        flush_async_sends()
        if output is not None:
            output.close()
        if trace is not None:
            trace.close()
        if runtime.owns_mpi_init and MPI is not None and MPI.Is_initialized() and not MPI.Is_finalized():
            comm.Barrier()
            MPI.Finalize()
    return TDStats(rank=rank, size=size)


def run_td_full_mpi_1d(
    r: np.ndarray,
    v: np.ndarray,
    mass: Union[float, np.ndarray],
    box: float,
    potential,
    dt: float,
    cutoff: float,
    n_steps: int,
    *,
    config: TDFullMPIRunConfig | None = None,
    output_spec: OutputSpec | None = None,
    **legacy_kwargs: Any,
):
    """TD full-MPI public entry point with compact config object.

    Backward compatibility:
      - legacy keyword options (thermo_every, zones_total, etc.) are still accepted;
      - when both ``config`` and legacy kwargs are provided, raises ``TypeError``.
    """
    if config is not None and legacy_kwargs:
        keys = ", ".join(sorted(legacy_kwargs.keys()))
        raise TypeError(
            f"run_td_full_mpi_1d received both config and legacy keyword options ({keys}); "
            "use one style"
        )
    cfg = config if config is not None else TDFullMPIRunConfig.from_legacy_kwargs(legacy_kwargs)
    return _run_td_full_mpi_1d_legacy(
        r=r,
        v=v,
        mass=mass,
        box=box,
        potential=potential,
        dt=dt,
        cutoff=cutoff,
        n_steps=n_steps,
        config=cfg,
        output_spec=output_spec,
    )
