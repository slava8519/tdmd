from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Union

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


def _run_td_full_mpi_1d_legacy(
    r: np.ndarray,
    v: np.ndarray,
    mass: Union[float, np.ndarray],
    box: float,
    potential,
    dt: float,
    cutoff: float,
    n_steps: int,
    thermo_every: int,
    cell_size: float,
    zones_total: int,
    zone_cells_w: int,
    zone_cells_s: int,
    zone_cells_pattern,
    traversal: str,
    fast_sync: bool,
    strict_fast_sync: bool,
    startup_mode: str,
    warmup_steps: int,
    warmup_compute: bool,
    buffer_k: float,
    use_verlet: bool,
    verlet_k_steps: int,
    skin_from_buffer: bool,
    formal_core: bool = True,
    batch_size: int = 4,
    overlap_mode: str = "table_support",
    debug_invariants: bool = False,
    strict_min_zone_width: bool = False,
    enable_step_id: bool = True,
    max_step_lag: int = 1,
    table_max_age: int = 1,
    max_pending_delta_atoms: int = 200000,
    require_local_deps: bool = True,
    require_table_deps: bool = True,
    require_owner_deps: bool = False,
    require_owner_ver: bool = True,
    enable_req_holder: bool = True,
    holder_gossip: bool = True,
    deps_provider_mode: str = "dynamic",
    zones_nx: int = 1,
    zones_ny: int = 1,
    zones_nz: int = 1,
    owner_buffer: float = 0.0,
    cuda_aware_mpi: bool = False,
    comm_overlap_isend: bool = False,
    atom_types: np.ndarray | None = None,
    ensemble_kind: str = "nve",
    thermostat: object | None = None,
    barostat: object | None = None,
    device: str = "cpu",
    output_spec: OutputSpec | None = None,
    trace_enabled: bool = False,
    trace_path: str = "td_trace.csv",
):
    if MPI is None:
        raise RuntimeError("mpi4py required")
    if not MPI.Is_initialized():
        MPI.Init()
    batch_size = int(batch_size)
    if batch_size < 1:
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

    atom_types = normalize_atom_types(atom_types, n_atoms=r.shape[0])

    trace = None
    if trace_enabled:
        trace_file = trace_path
        if size > 1:
            base, ext = os.path.splitext(trace_path)
            if rank != 0:
                trace_file = f"{base}_rank{rank}{ext or '.csv'}"
        trace = TDTraceLogger(trace_file, rank=rank, enabled=True)

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

    deps_mode = str(deps_provider_mode)
    static_rr = deps_mode == "static_rr"
    if static_rr and startup_mode != "scatter_zones":
        raise RuntimeError("static_rr requires startup_mode=scatter_zones (static ownership)")

    if strict_fast_sync:
        if not (fast_sync and zones_total == 2 * size and startup_mode == "scatter_zones"):
            raise RuntimeError(
                "strict_fast_sync requires fast_sync=true, zones_total=2P, startup_mode=scatter_zones"
            )

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

    require_table_deps = bool(require_table_deps or require_local_deps)

    def _trace_event(
        *,
        zid: int,
        step_id: int,
        event: str,
        state_before=None,
        state_after=None,
        halo_ids_count: int = 0,
        migration_count: int = 0,
        lag: int = 0,
    ):
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

    def _lag_value(zid: int) -> int:
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

    def _start_compute_with_trace(
        r: np.ndarray, rc: float, skin_global: float, step: int, verlet_k_steps: int
    ):
        pre = [z.ztype for z in zones] if trace is not None else None
        zid = autom.start_compute(
            r=r, rc=rc, skin_global=skin_global, step=step, verlet_k_steps=verlet_k_steps
        )
        if zid is not None and trace is not None and pre is not None:
            _trace_event(
                zid=zid,
                step_id=int(zones[zid].step_id),
                event="START_COMPUTE",
                state_before=pre[zid],
                state_after=zones[zid].ztype,
                halo_ids_count=int(zones[zid].halo_ids.size),
                migration_count=0,
                lag=_lag_value(zid),
            )
            for did in autom._locked_donors.get(zid, []):
                _trace_event(
                    zid=did,
                    step_id=int(zones[did].step_id),
                    event="LOCK_DONOR",
                    state_before=pre[did],
                    state_after=zones[did].ztype,
                    halo_ids_count=int(zones[did].halo_ids.size),
                    migration_count=0,
                    lag=_lag_value(did),
                )
        return zid

    def _finish_compute_with_trace(
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
            _trace_event(
                zid=zid,
                step_id=int(zones[zid].step_id),
                event="FINISH_COMPUTE",
                state_before=pre[zid],
                state_after=zones[zid].ztype,
                halo_ids_count=0,
                migration_count=int(autom.diag.get("migrations", 0)),
                lag=_lag_value(zid),
            )
            for did, pstate in enumerate(pre):
                if pstate == ZoneType.P and zones[did].ztype == ZoneType.D:
                    _trace_event(
                        zid=did,
                        step_id=int(zones[did].step_id),
                        event="RELEASE_DONOR",
                        state_before=pstate,
                        state_after=zones[did].ztype,
                        halo_ids_count=int(zones[did].halo_ids.size),
                        migration_count=0,
                        lag=_lag_value(did),
                    )
        return zid

    deps_provider_3d = None
    if deps_mode == "static_3d":
        deps_provider_3d = DepsProvider3DBlock(
            nx=int(zones_nx),
            ny=int(zones_ny),
            nz=int(zones_nz),
            box=(float(box), float(box), float(box)),
            cutoff=float(cutoff),
            mpi_size=int(size),
        )

        class _GeomProvider:
            def __init__(self, base):
                self.base = base

            def geom(self, zid: int) -> ZoneGeomAABB:
                lo, hi = self.base.geom(int(zid))
                return ZoneGeomAABB(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))

        autom.geom_provider = _GeomProvider(deps_provider_3d)

    def _geom_aabb(zid: int) -> ZoneGeomAABB:
        if deps_provider_3d is None:
            raise RuntimeError("static_3d required for AABB geometry")
        lo, hi = deps_provider_3d.geom(int(zid))
        return ZoneGeomAABB(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))

    def deps_zone_ids(zid: int) -> list[int]:
        if deps_provider_3d is not None:
            return [int(d) for d in deps_provider_3d.deps_table(int(zid))]
        z = zones[int(zid)]
        deps = zones_overlapping_range_pbc(z.z0 - cutoff, z.z1 + cutoff, box, zones)
        return [int(d) for d in deps if int(d) != int(zid)]

    def owner_deps_zone_ids(zid: int) -> list[int]:
        if deps_provider_3d is not None:
            return [int(d) for d in deps_provider_3d.deps_owner(int(zid))]
        z = zones[int(zid)]
        buf = float(owner_buffer) if owner_buffer and owner_buffer > 0 else float(cutoff)
        if owner_buffer <= 0:
            buf = max(buf, float(getattr(z, "buffer", 0.0)))
        deps = zones_overlapping_range_pbc(z.z0 - buf, z.z1 + buf, box, zones)
        return [int(d) for d in deps if int(d) != int(zid)]

    # v2.9: dynamic holder map for direct routing of HALO/DELTA
    # holder_map[zid] = rank that currently holds (non-F) state for that zone.
    if static_rr:
        holder_map = [int(zid) % int(size) for zid in range(zones_total)]
        holder_ver = [0 for _ in range(zones_total)]
        holder_epoch = 0
    else:
        # Initialized once using allgather; then updated locally on REC_ZONE sends/receives.
        local_have = [z.zid for z in zones if z.ztype != ZoneType.F]
        all_have = comm.allgather(local_have)
        holder_map = [-1 for _ in range(zones_total)]
        holder_ver = [-1 for _ in range(zones_total)]
        holder_epoch = 0
        for rrk, zlist in enumerate(all_have):
            for zid in zlist:
                holder_map[int(zid)] = int(rrk)
                holder_ver[int(zid)] = 0

    # deps_table: all zones intersecting p-neighborhood within cutoff
    autom.set_deps_funcs(
        table_func=lambda zid: deps_zone_ids(int(zid)),
        owner_func=lambda zid: owner_deps_zone_ids(int(zid)),
    )
    if static_rr and require_owner_deps:
        owner_pred = lambda did: holder_map[int(did)] != -1
    else:
        owner_pred = lambda did: (
            (
                (holder_map[int(did)] != -1)
                and (
                    (not require_owner_ver)
                    or (holder_ver[int(did)] >= (holder_epoch - (2 * max_step_lag + 2)))
                )
            )
            if require_owner_deps
            else True
        )
    autom.set_deps_preds(
        table_pred=(
            lambda did: (zones[int(did)].ztype != ZoneType.F) if require_table_deps else True
        ),
        owner_pred=owner_pred,
    )

    req_reply_outbox: dict[int, list[tuple[int, int, int, np.ndarray, int]]] = (
        {}
    )  # dest_rank -> records
    holder_reply_outbox: dict[int, list[tuple[int, int, int, np.ndarray, int]]] = (
        {}
    )  # dest_rank -> holder gossip replies

    diag = autom.diag

    def _set_holder(zid: int, rk: int):
        if static_rr:
            return
        holder_map[int(zid)] = int(rk)

    skin_global = 0.0

    def update_buffers(step: int):
        nonlocal skin_global
        skin_global = 0.0
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
            # diagnostic: buffer sufficiency for linear drift
            if max_step_lag > 0:
                speeds = np.linalg.norm(v[z.atom_ids], axis=1)
                vmax = float(speeds.max()) if speeds.size else 0.0
                required = vmax * dt * float(max_step_lag + 1)
                if b + FLOAT_EQ_ATOL < required:
                    autom.diag["viol_buffer"] = autom.diag.get("viol_buffer", 0) + 1
            skin_global = max(skin_global, sk)

    def _scale_zone_geometry(lam: float, new_box: float) -> None:
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

    def _apply_ensemble(step: int):
        nonlocal box
        new_box, _lam_t, lam_b = apply_ensemble_step(
            step=int(step),
            ensemble=ensemble,
            r=r,
            v=v,
            mass=mass,
            box=float(box),
            potential=potential,
            cutoff=cutoff,
            atom_types=atom_types,
            dt=dt,
        )
        if ensemble.kind == "npt":
            _scale_zone_geometry(lam_b, new_box)
            box = float(new_box)

    def _output_due(step: int):
        if not output_enabled:
            return False, False, False
        due_traj = (out_traj_every > 0) and (step % out_traj_every == 0)
        due_metrics = (out_metrics_every > 0) and (step % out_metrics_every == 0)
        return (due_traj or due_metrics), due_traj, due_metrics

    def _gather_for_output():
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

    pending_deltas: dict[tuple[int, int, int], list[np.ndarray]] = (
        {}
    )  # (zid, step_id, subtype) -> list of ids arrays
    pending_atoms: int = 0

    def _pending_size() -> int:
        return pending_atoms

    def _validate_halo_geometry(zid: int, halo_ids: np.ndarray):
        if halo_ids.size == 0:
            return
        z = zones[zid]
        z0p = z.z0 - cutoff
        z1p = z.z1 + cutoff
        mask = _within_interval_pbc(r[halo_ids, 2], z0p, z1p, box)
        bad = int((~mask).sum())
        if bad > 0:
            autom.diag['halo_geo_viol'] = autom.diag.get('halo_geo_viol', 0) + bad
        # optional support-invariant: if table exists, halo should lie in support(table(zone))
        if getattr(z, 'table', None) is not None and z.table is not None:
            supp = set(map(int, z.table.support_ids().tolist()))
            viol = sum(1 for a in halo_ids.tolist() if int(a) not in supp)
            if viol:
                autom.diag['halo_support_viol'] = autom.diag.get('halo_support_viol', 0) + int(viol)

    def _cleanup_pending():
        """Bound pending delta buffer: drop too-old and overflow."""
        nonlocal pending_atoms
        to_drop = []
        for (zid, sid, st), parts in pending_deltas.items():
            z = zones[zid]
            if z.ztype != ZoneType.F and (int(z.step_id) - int(sid)) > max_step_lag:
                to_drop.append((zid, sid, st))
        for key in to_drop:
            parts = pending_deltas.pop(key, [])
            dropped = int(sum(p.size for p in parts))
            pending_atoms -= dropped
            autom.diag['delta_dropped'] = autom.diag.get('delta_dropped', 0) + dropped
        if max_pending_delta_atoms <= 0:
            return
        while pending_atoms > max_pending_delta_atoms and pending_deltas:
            oldest = min(pending_deltas.keys(), key=lambda k: k[1])
            parts = pending_deltas.pop(oldest, [])
            dropped = int(sum(p.size for p in parts))
            pending_atoms -= dropped
            autom.diag['delta_dropped_overflow'] = (
                autom.diag.get('delta_dropped_overflow', 0) + dropped
            )
            autom.diag['delta_dropped'] = autom.diag.get('delta_dropped', 0) + dropped

    def _apply_pending_if_ready(zid: int):
        nonlocal pending_atoms
        z = zones[zid]
        keys = [k for k in pending_deltas.keys() if k[0] == zid and k[1] == int(z.step_id)]
        if not keys:
            return
        for k in keys:
            _, _, st = k
            parts = pending_deltas.pop(k)
            if not parts:
                continue
            add = (
                np.concatenate(parts).astype(np.int32)
                if len(parts) > 1
                else parts[0].astype(np.int32)
            )
            pending_atoms -= int(add.size)
            if add.size == 0:
                continue
            if int(st) == DELTA_HALO:
                if int(z.halo_step_id) != int(z.step_id):
                    z.halo_ids = np.unique(add).astype(np.int32)
                    # v4.3: static_3d halo geometric check vs receiver AABB (I4, direct)
                    if str(deps_provider_mode) == 'static_3d' and z.halo_ids.size > 0:
                        gZ = _geom_aabb(int(z.zid))
                        m = mask_in_aabb_pbc(
                            r, z.halo_ids.astype(np.int32), gZ.lo, gZ.hi, float(cutoff), float(box)
                        )
                        if not bool(m.all()):
                            diag['hG3'] = diag.get('hG3', 0) + int((~m).sum())
                    z.halo_step_id = int(z.step_id)
                else:
                    z.halo_ids = np.unique(np.concatenate([z.halo_ids, add]).astype(np.int32))
                    # v4.3: static_3d halo geometric check vs receiver AABB (I4, direct)
                    if str(deps_provider_mode) == 'static_3d' and z.halo_ids.size > 0:
                        gZ = _geom_aabb(int(z.zid))
                        m = mask_in_aabb_pbc(
                            r, z.halo_ids.astype(np.int32), gZ.lo, gZ.hi, float(cutoff), float(box)
                        )
                        if not bool(m.all()):
                            diag['hG3'] = diag.get('hG3', 0) + int((~m).sum())
                _validate_halo_geometry(zid, z.halo_ids)
                autom.diag["halo_applied"] = autom.diag.get("halo_applied", 0) + int(add.size)
            else:
                z.atom_ids = (
                    add
                    if z.atom_ids.size == 0
                    else np.concatenate([z.atom_ids, add]).astype(np.int32)
                )
                if z.ztype == ZoneType.F:
                    z.ztype = ZoneType.D
                autom.diag["delta_applied"] = autom.diag.get("delta_applied", 0) + int(add.size)

    def _handle_record(
        src_rank: int,
        rectype: int,
        subtype: int,
        zid: int,
        ids: np.ndarray,
        rr: np.ndarray,
        vv: np.ndarray,
        step_id: int,
    ):
        nonlocal pending_atoms
        state_before = zones[zid].ztype if 0 <= int(zid) < len(zones) else None
        if ids.size:
            r[ids] = rr
            v[ids] = vv

        if rectype == REC_HOLDER:
            # holder gossip update: zid is held by src_rank with version=step_id
            ver = int(step_id)
            if ver > int(holder_ver[int(zid)]):
                holder_map[int(zid)] = int(src_rank)
                holder_ver[int(zid)] = ver
            return

        if rectype == REC_REQ:
            # Request from src_rank.
            # subtype:
            #   REQ_HALO   (0): request HALO for dependency zid
            #   REQ_HOLDER (1): request holder-map info for dependency zid (respond with REC_HOLDER)
            autom.diag["req_rcv"] = autom.diag.get("req_rcv", 0) + 1
            dep_zid = int(zid)

            if int(subtype) == REQ_HOLDER:
                # Respond with current holder knowledge for dep_zid.
                ver = int(holder_ver[dep_zid]) if 0 <= dep_zid < zones_total else -1
                rec = (REC_HOLDER, 0, dep_zid, np.empty((0,), np.int32), ver)
                holder_reply_outbox.setdefault(int(src_rank), []).append(rec)
                autom.diag["req_holder_reply"] = autom.diag.get("req_holder_reply", 0) + 1
                return

            # REQ_HALO
            if ids.size >= 2:
                req_zid = int(ids[0])
                _req_step = int(ids[1])  # noqa: F841 – reserved for future lag check
            else:
                req_zid = -1
                _req_step = -1  # noqa: F841

            if (
                0 <= dep_zid < zones_total
                and zones[dep_zid].ztype != ZoneType.F
                and zones[dep_zid].atom_ids.size
            ):
                if 0 <= req_zid < zones_total:
                    if str(deps_provider_mode) == 'static_3d':
                        g = _geom_aabb(int(req_zid))
                        halo_ids = halo_filter_by_receiver_aabb(
                            r, zones[dep_zid].atom_ids.astype(np.int32), g.lo, g.hi, cutoff, box
                        )
                    else:
                        halo_ids = halo_filter_by_receiver_p(
                            r,
                            zones[dep_zid].atom_ids.astype(np.int32),
                            zones[req_zid].z0,
                            zones[req_zid].z1,
                            cutoff,
                            box,
                        )
                else:
                    halo_ids = zones[dep_zid].atom_ids.astype(np.int32)
                if halo_ids.size:
                    rec = (
                        REC_DELTA,
                        DELTA_HALO,
                        dep_zid,
                        halo_ids.astype(np.int32),
                        int(zones[dep_zid].step_id),
                    )
                    req_reply_outbox.setdefault(int(src_rank), []).append(rec)
                    autom.diag["req_reply_sent"] = autom.diag.get("req_reply_sent", 0) + 1
            return

        if rectype == REC_ZONE:
            autom.on_recv(zid, ids, step_id)
            _trace_event(
                zid=zid,
                step_id=int(step_id),
                event="RECV_ZONE",
                state_before=state_before,
                state_after=zones[zid].ztype,
                halo_ids_count=0,
                migration_count=int(ids.size),
                lag=_lag_value(zid),
            )
            _cleanup_pending()
            _apply_pending_if_ready(zid)
            return

        # DELTA
        sid = int(step_id)
        z = zones[zid]
        if int(subtype) == DELTA_HALO and z.ztype == ZoneType.F:
            # create shadow dependency zone carrying only HALO
            z.ztype = ZoneType.P
            z.step_id = sid
            z.halo_step_id = sid
            z.halo_ids = np.empty((0,), np.int32)
            # v4.3: static_3d halo geometric check vs receiver AABB (I4, direct)
            if str(deps_provider_mode) == 'static_3d' and z.halo_ids.size > 0:
                gZ = _geom_aabb(int(z.zid))
                m = mask_in_aabb_pbc(
                    r, z.halo_ids.astype(np.int32), gZ.lo, gZ.hi, float(cutoff), float(box)
                )
                if not bool(m.all()):
                    diag['hG3'] = diag.get('hG3', 0) + int((~m).sum())
            autom.diag['shadow_promoted'] = autom.diag.get('shadow_promoted', 0) + 1

        if z.ztype != ZoneType.F and int(z.step_id) == sid:
            if ids.size:
                if int(subtype) == DELTA_HALO:
                    if int(z.halo_step_id) != int(z.step_id):
                        z.halo_ids = np.unique(ids).astype(np.int32)
                        # v4.3: static_3d halo geometric check vs receiver AABB (I4, direct)
                        if str(deps_provider_mode) == 'static_3d' and z.halo_ids.size > 0:
                            gZ = _geom_aabb(int(z.zid))
                            m = mask_in_aabb_pbc(
                                r,
                                z.halo_ids.astype(np.int32),
                                gZ.lo,
                                gZ.hi,
                                float(cutoff),
                                float(box),
                            )
                            if not bool(m.all()):
                                diag['hG3'] = diag.get('hG3', 0) + int((~m).sum())
                        z.halo_step_id = int(z.step_id)
                    else:
                        z.halo_ids = np.unique(np.concatenate([z.halo_ids, ids]).astype(np.int32))
                        # v4.3: static_3d halo geometric check vs receiver AABB (I4, direct)
                        if str(deps_provider_mode) == 'static_3d' and z.halo_ids.size > 0:
                            gZ = _geom_aabb(int(z.zid))
                            m = mask_in_aabb_pbc(
                                r,
                                z.halo_ids.astype(np.int32),
                                gZ.lo,
                                gZ.hi,
                                float(cutoff),
                                float(box),
                            )
                            if not bool(m.all()):
                                diag['hG3'] = diag.get('hG3', 0) + int((~m).sum())
                    _validate_halo_geometry(zid, z.halo_ids)
                    autom.diag["halo_applied"] = autom.diag.get("halo_applied", 0) + int(ids.size)
                    _trace_event(
                        zid=zid,
                        step_id=int(sid),
                        event="RECV_DELTA_HALO",
                        state_before=state_before,
                        state_after=z.ztype,
                        halo_ids_count=int(ids.size),
                        migration_count=0,
                        lag=_lag_value(zid),
                    )
                else:
                    z.atom_ids = (
                        ids
                        if z.atom_ids.size == 0
                        else np.concatenate([z.atom_ids, ids]).astype(np.int32)
                    )
                    autom.diag["delta_applied"] = autom.diag.get("delta_applied", 0) + int(ids.size)
                    _trace_event(
                        zid=zid,
                        step_id=int(sid),
                        event="RECV_DELTA_MIGRATION",
                        state_before=state_before,
                        state_after=z.ztype,
                        halo_ids_count=0,
                        migration_count=int(ids.size),
                        lag=_lag_value(zid),
                    )
            return

        if z.ztype != ZoneType.F and (int(z.step_id) - sid) > max_step_lag:
            if int(subtype) == DELTA_HALO:
                autom.diag["halo_dropped"] = autom.diag.get("halo_dropped", 0) + int(ids.size)
            autom.diag["delta_dropped"] = autom.diag.get("delta_dropped", 0) + int(ids.size)
            return

        pending_deltas.setdefault((zid, sid, int(subtype)), []).append(ids.astype(np.int32))
        pending_atoms += int(ids.size)
        _cleanup_pending()
        if int(subtype) == DELTA_HALO:
            autom.diag["halo_deferred"] = autom.diag.get("halo_deferred", 0) + int(ids.size)
        else:
            autom.diag["delta_deferred"] = autom.diag.get("delta_deferred", 0) + int(ids.size)

    def recv_phase(tag_base: int):
        # Forward direction: from prev_rank on tag_base
        while comm.Iprobe(source=prev_rank, tag=tag_base):
            payload = recv_payload(comm, prev_rank, tag=tag_base)
            for rectype, subtype, zid, ids, rr, vv, step_id in unpack_records(payload):
                _handle_record(prev_rank, rectype, subtype, zid, ids, rr, vv, step_id)

        # Backward direction halos: from next_rank on tag_base+2
        while comm.Iprobe(source=next_rank, tag=tag_base + 2):
            payload = recv_payload(comm, next_rank, tag=tag_base + 2)
            for rectype, subtype, zid, ids, rr, vv, step_id in unpack_records(payload):
                _handle_record(next_rank, rectype, subtype, zid, ids, rr, vv, step_id)

    def overlap_set_for_send(zid: int, next_zid: int, rc: float, step: int) -> set:
        if zones[zid].atom_ids.size == 0:
            return set()
        if overlap_mode == "geometric_rc":
            if str(deps_provider_mode) == 'static_3d':
                # 3D: compute overlap via receiver AABB+cutoff (support region)
                gR = _geom_aabb(int(next_zid))
                ov = overlap_filter_by_receiver_aabb(
                    r, zones[zid].atom_ids.astype(np.int32), gR, rc, box
                )
                return set(map(int, ov.tolist()))
            ov = halo_filter_by_receiver_p(
                r,
                zones[zid].atom_ids.astype(np.int32),
                zones[next_zid].z0,
                zones[next_zid].z1,
                rc,
                box,
            )
            return set(map(int, ov.tolist()))

        autom.ensure_table(
            next_zid,
            r=r,
            rc=rc,
            skin_global=(skin_global if use_verlet else 0.0),
            step=step,
            verlet_k_steps=(verlet_k_steps if use_verlet else 1),
        )
        support = autom.table_support(next_zid)
        if support.size == 0:
            return set()
        return set(
            map(int, np.intersect1d(zones[zid].atom_ids, support, assume_unique=False).tolist())
        )

    def send_phase(tag_base: int, rc: float, step: int):
        # Build per-destination record buckets for direct routing
        buckets: dict[int, list[tuple[int, int, int, np.ndarray, int]]] = {}

        # flush pending replies to REQ
        if req_reply_outbox:
            for dr, recs in list(req_reply_outbox.items()):
                if recs:
                    buckets.setdefault(int(dr), []).extend(recs)
            req_reply_outbox.clear()
        if holder_reply_outbox:
            for dr, recs in list(holder_reply_outbox.items()):
                if recs:
                    buckets.setdefault(int(dr), []).extend(recs)
            holder_reply_outbox.clear()

        def add_record(dest_rank: int, rec):
            buckets.setdefault(int(dest_rank), []).append(rec)

        batch = autom.pop_send_batch(batch_size=batch_size)
        while batch:
            for zid in batch:
                z = zones[zid]
                if z.ztype != ZoneType.S:
                    continue
                z_state_before = z.ztype

                # deps-based halo: send to current holders of deps zones (direct)
                deps = deps_zone_ids(zid)
                for dep_zid in deps:
                    try:
                        ov = overlap_set_for_send(zid, dep_zid, rc=rc, step=step)
                    except (KeyError, IndexError, ValueError):
                        ov = set()
                    ov_ids = (
                        np.array(list(ov), dtype=np.int32) if ov else z.atom_ids.astype(np.int32)
                    )
                    if str(deps_provider_mode) == 'static_3d':
                        g = _geom_aabb(int(dep_zid))
                        halo_ids = halo_filter_by_receiver_aabb(r, ov_ids, g.lo, g.hi, cutoff, box)
                    else:
                        halo_ids = halo_filter_by_receiver_p(
                            r, ov_ids, zones[dep_zid].z0, zones[dep_zid].z1, cutoff, box
                        )
                    if halo_ids.size:
                        dest = holder_map[int(dep_zid)]
                        if dest < 0:
                            dest = next_rank  # fallback
                        add_record(
                            dest, (REC_DELTA, DELTA_HALO, int(dep_zid), halo_ids, int(z.step_id))
                        )
                        autom.diag['halo_sent'] = autom.diag.get('halo_sent', 0) + int(
                            halo_ids.size
                        )
                        _trace_event(
                            zid=zid,
                            step_id=int(z.step_id),
                            event="SEND_DELTA_HALO",
                            state_before=z_state_before,
                            state_after=z_state_before,
                            halo_ids_count=int(halo_ids.size),
                            migration_count=0,
                            lag=_lag_value(zid),
                        )

                if static_rr:
                    # Static ownership: keep zone local, only halos/deltas are routed.
                    z.ztype = ZoneType.D if z.atom_ids.size else ZoneType.F
                    _trace_event(
                        zid=zid,
                        step_id=int(z.step_id),
                        event="SEND_ZONE",
                        state_before=z_state_before,
                        state_after=z.ztype,
                        halo_ids_count=0,
                        migration_count=0,
                        lag=_lag_value(zid),
                    )
                else:
                    # Ownership pipeline: computed zone state goes to next_rank (as before)
                    next_zid = (zid + 1) % zones_total
                    overlap_next = overlap_set_for_send(zid, next_zid, rc=rc, step=step)
                    if overlap_next:
                        mask_send = np.array(
                            [int(a) not in overlap_next for a in z.atom_ids], dtype=bool
                        )
                        send_ids = z.atom_ids[mask_send].astype(np.int32)
                        keep_ids = z.atom_ids[~mask_send].astype(np.int32)
                    else:
                        send_ids = z.atom_ids.astype(np.int32)
                        keep_ids = np.empty((0,), np.int32)

                    if debug_invariants and overlap_next:
                        if any(int(a) in overlap_next for a in send_ids.tolist()):
                            autom.diag["viol_send_overlap"] += 1

                    # Send zone state to next_rank (pipeline)
                    add_record(next_rank, (REC_ZONE, 0, int(zid), send_ids, int(z.step_id)))
                    _set_holder(zid, next_rank)

                    # commit local keep
                    z.atom_ids = keep_ids
                    z.table = None
                    z.ztype = ZoneType.D if keep_ids.size else ZoneType.F
                    _trace_event(
                        zid=zid,
                        step_id=int(z.step_id),
                        event="SEND_ZONE",
                        state_before=z_state_before,
                        state_after=z.ztype,
                        halo_ids_count=0,
                        migration_count=int(send_ids.size),
                        lag=_lag_value(zid),
                    )

            batch = autom.pop_send_batch(batch_size=batch_size)

        # outbox (migration) deltas: route to current holder of destination zone (direct)
        for dest_zid, sid, ids in autom.iter_outbox_records():
            if ids.size:
                dest = holder_map[int(dest_zid)]
                if dest < 0:
                    dest = next_rank
                add_record(dest, (REC_DELTA, DELTA_MIGRATION, int(dest_zid), ids, int(sid)))
                _trace_event(
                    zid=dest_zid,
                    step_id=int(sid),
                    event="SEND_DELTA_MIGRATION",
                    state_before=(
                        zones[dest_zid].ztype if 0 <= int(dest_zid) < zones_total else None
                    ),
                    state_after=zones[dest_zid].ztype if 0 <= int(dest_zid) < zones_total else None,
                    halo_ids_count=0,
                    migration_count=int(ids.size),
                    lag=_lag_value(dest_zid) if 0 <= int(dest_zid) < zones_total else 0,
                )
        autom.clear_outbox()

        # flush all buckets
        send_reqs = []
        send_keepalive = []
        for dest_rank, recs in buckets.items():
            if not recs:
                continue
            payload = pack_records(recs, r, v)
            if use_async_send:
                reqs, refs = post_send_payload(comm, int(dest_rank), tag=tag_base, payload=payload)
                send_reqs.extend(reqs)
                send_keepalive.extend(refs)
                autom.diag["async_send_msgs"] = autom.diag.get("async_send_msgs", 0) + 1
                autom.diag["async_send_bytes"] = autom.diag.get("async_send_bytes", 0) + int(
                    len(payload)
                )
            else:
                send_payload(comm, int(dest_rank), tag=tag_base, payload=payload)
        if send_reqs:
            MPI.Request.Waitall(send_reqs)
            send_keepalive.clear()

    warmup_steps = max(0, int(warmup_steps))
    if warmup_steps > 0 and rank == 0:
        print(
            f"[info] warmup: steps={warmup_steps} compute={warmup_compute} formal_core={formal_core} "
            f"batch_size={batch_size} overlap_mode={overlap_mode} step_id={enable_step_id} "
            f"max_lag={max_step_lag} table_max_age={table_max_age}",
            flush=True,
        )

    if output_enabled:
        update_buffers(step=0)
        due, due_traj, due_metrics = _output_due(0)
        if due:
            r_all, v_all, bmean = _gather_for_output()
            if rank == 0 and output is not None and r_all is not None and v_all is not None:
                if due_traj and output.traj is not None:
                    output.traj.write(
                        0, r_all, v_all, box_value=(float(box), float(box), float(box))
                    )
                if due_metrics and output.metrics is not None:
                    output.metrics.write(0, r_all, v_all, buffer_value=bmean, box_value=float(box))

    for w in range(1, warmup_steps + 1):
        update_buffers(step=w)
        rc = cutoff + skin_global
        if (rank % 2) == 0:
            recv_phase(tag_base=7000 + 10 * w)
            if warmup_compute:
                if autom.can_start_compute():
                    _start_compute_with_trace(
                        r=r,
                        rc=rc,
                        skin_global=(skin_global if use_verlet else 0.0),
                        step=w,
                        verlet_k_steps=(verlet_k_steps if use_verlet else 1),
                    )
                _finish_compute_with_trace(
                    r=r,
                    v=v,
                    mass=mass,
                    dt=dt,
                    potential=potential,
                    cutoff=cutoff,
                    rc=rc,
                    skin_global=(skin_global if use_verlet else 0.0),
                    step=w,
                    verlet_k_steps=(verlet_k_steps if use_verlet else 1),
                    enable_step_id=enable_step_id,
                )
            send_phase(tag_base=7000 + 10 * w, rc=rc, step=w)
        else:
            send_phase(tag_base=7000 + 10 * w, rc=rc, step=w)
            recv_phase(tag_base=7000 + 10 * w)
        comm.Barrier()

    for step in range(1, n_steps + 1):
        s = 1000 + step
        update_buffers(step=s)
        rc = cutoff + skin_global

        if (rank % 2) == 0:
            recv_phase(tag_base=6000)

        if autom.can_start_compute():
            _start_compute_with_trace(
                r=r,
                rc=rc,
                skin_global=(skin_global if use_verlet else 0.0),
                step=s,
                verlet_k_steps=(verlet_k_steps if use_verlet else 1),
            )
        _finish_compute_with_trace(
            r=r,
            v=v,
            mass=mass,
            dt=dt,
            potential=potential,
            cutoff=cutoff,
            rc=rc,
            skin_global=(skin_global if use_verlet else 0.0),
            step=s,
            verlet_k_steps=(verlet_k_steps if use_verlet else 1),
            enable_step_id=enable_step_id,
        )

        send_phase(tag_base=6000, rc=rc, step=s)

        if (rank % 2) != 0:
            recv_phase(tag_base=6000)

        _apply_ensemble(step)

        if output_enabled:
            due, due_traj, due_metrics = _output_due(step)
            if due:
                r_all, v_all, bmean = _gather_for_output()
                if rank == 0 and output is not None and r_all is not None and v_all is not None:
                    if due_traj and output.traj is not None:
                        output.traj.write(
                            step, r_all, v_all, box_value=(float(box), float(box), float(box))
                        )
                    if due_metrics and output.metrics is not None:
                        output.metrics.write(
                            step, r_all, v_all, buffer_value=bmean, box_value=float(box)
                        )

        if thermo_every and (step % thermo_every == 0) and rank == 0:
            ke = kinetic_energy(v, mass)
            T = temperature_from_ke(ke, r.shape[0])
            bmean = float(np.mean([zz.buffer for zz in zones]))
            q = len(autom.send_queue)
            # Local WFG diagnostics (no global sync)
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
                f"[step={step}] T={T:.3f} skin={skin_global:.4f} <b>={bmean:.4f} sendQ={q} rc={rc:.4f} "
                f"mig={diag['migrations']} out={diag.get('outbox_atoms',0)} lagV={diag['viol_lag']} bufV={diag.get('viol_buffer',0)} dA={diag.get('delta_applied',0)} dD={diag.get('delta_deferred',0)} hS={diag.get('halo_sent',0)} hA={diag.get('halo_applied',0)} hD={diag.get('halo_deferred',0)} hG={diag.get('halo_geo_viol',0)} hV={diag.get('halo_support_viol',0)} dX={diag.get('delta_dropped',0)} hX={diag.get('halo_dropped',0)} dO={diag.get('delta_dropped_overflow',0)} "
                f"tblAgeReb={diag['table_rebuild_age']} violW={diag['viol_w_gt1']} violO={diag['viol_send_overlap']} "
                f"zone_step_spread={spread} reqS={diag.get('req_sent',0)} reqR={diag.get('req_rcv',0)} reqHS={diag.get('req_holder_sent',0)} reqHR={diag.get('req_holder_reply',0)} sh={diag.get('shadow_promoted',0)} wT={diag.get('wait_table',0)} wO={diag.get('wait_owner',0)} wOU={diag.get('wait_owner_unknown',0)} wOS={diag.get('wait_owner_stale',0)} "
                f"tbK={batch_size} sb={sb} sbAvg={sb_avg:.2f} sbMax={sb_max} asyncS={diag.get('async_send_msgs',0)} asyncB={diag.get('async_send_bytes',0)} wfgS={diag.get('wfg_samples',0)} wfgC={diag.get('wfg_cycles',0)} wfgO={diag.get('wfg_max_outdeg',0)}",
                flush=True,
            )

    if output is not None:
        output.close()
    if trace is not None:
        trace.close()
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
        thermo_every=cfg.thermo_every,
        cell_size=cfg.cell_size,
        zones_total=cfg.zones_total,
        zone_cells_w=cfg.zone_cells_w,
        zone_cells_s=cfg.zone_cells_s,
        zone_cells_pattern=cfg.zone_cells_pattern,
        traversal=cfg.traversal,
        fast_sync=cfg.fast_sync,
        strict_fast_sync=cfg.strict_fast_sync,
        startup_mode=cfg.startup_mode,
        warmup_steps=cfg.warmup_steps,
        warmup_compute=cfg.warmup_compute,
        buffer_k=cfg.buffer_k,
        use_verlet=cfg.use_verlet,
        verlet_k_steps=cfg.verlet_k_steps,
        skin_from_buffer=cfg.skin_from_buffer,
        formal_core=cfg.formal_core,
        batch_size=cfg.batch_size,
        overlap_mode=cfg.overlap_mode,
        debug_invariants=cfg.debug_invariants,
        strict_min_zone_width=cfg.strict_min_zone_width,
        enable_step_id=cfg.enable_step_id,
        max_step_lag=cfg.max_step_lag,
        table_max_age=cfg.table_max_age,
        max_pending_delta_atoms=cfg.max_pending_delta_atoms,
        require_local_deps=cfg.require_local_deps,
        require_table_deps=cfg.require_table_deps,
        require_owner_deps=cfg.require_owner_deps,
        require_owner_ver=cfg.require_owner_ver,
        enable_req_holder=cfg.enable_req_holder,
        holder_gossip=cfg.holder_gossip,
        deps_provider_mode=cfg.deps_provider_mode,
        zones_nx=cfg.zones_nx,
        zones_ny=cfg.zones_ny,
        zones_nz=cfg.zones_nz,
        owner_buffer=cfg.owner_buffer,
        cuda_aware_mpi=cfg.cuda_aware_mpi,
        comm_overlap_isend=cfg.comm_overlap_isend,
        atom_types=cfg.atom_types,
        ensemble_kind=cfg.ensemble_kind,
        thermostat=cfg.thermostat,
        barostat=cfg.barostat,
        device=cfg.device,
        output_spec=output_spec,
        trace_enabled=cfg.trace_enabled,
        trace_path=cfg.trace_path,
    )
