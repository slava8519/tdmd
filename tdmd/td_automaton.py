from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .celllist import CellList, build_cell_list
from .forces_cells import forces_on_targets_celllist_compact, forces_on_targets_zonecells
from .geom_pbc import mask_in_aabb_pbc
from .integrator import vv_finish_velocities, vv_update_positions
from .interaction_table_state import InteractionTableState
from .zone_bins_localz import PersistentZoneLocalZBinsCache
from .zones import ZoneType, zones_overlapping_range_pbc


@dataclass
class ZoneRuntime:
    zid: int
    z0: float
    z1: float
    ztype: ZoneType
    atom_ids: np.ndarray
    n_cells: int = 1
    halo_ids: np.ndarray = field(
        default_factory=lambda: np.empty((0,), np.int32)
    )  # ghosts/halo for interaction tables
    halo_step_id: int = -1  # zone-time layer for which halo_ids are valid
    step_id: int = 0
    buffer: float = 0.0
    skin: float = 0.0
    table: InteractionTableState | None = None


class TDAutomaton1W:
    """Строгое ядро TD: максимум одна зона в W на ранге.

    v1.8:
    - time-lag safety: запрещаем старт вычисления зоны, если зависимости слишком отстают по времени.
    - age-limited tables: перестраиваем таблицу, если она слишком стара по зоне-времени.
    """

    def __init__(
        self,
        zones_runtime: List[ZoneRuntime],
        box: float,
        cutoff: float,
        bins_cache: PersistentZoneLocalZBinsCache,
        traversal_order: List[int],
        formal_core: bool = True,
        debug_invariants: bool = False,
        max_step_lag: int = 1,
        table_max_age: int = 1,
    ):
        self.zones = zones_runtime
        self.box = float(box)
        self.cutoff = float(cutoff)
        self.cache = bins_cache
        self.order = list(traversal_order)
        self.formal_core = bool(formal_core)
        self.debug_invariants = bool(debug_invariants)

        self.max_step_lag = max(0, int(max_step_lag))
        self.table_max_age = max(0, int(table_max_age))
        # Optional geometry provider for 3D AABB support
        self.geom_provider = None

        self.zwidth = self.box / max(1, len(self.zones))

        self.work_zid: Optional[int] = None
        self.priority_key = None  # A4b: keyfunc(zid)->tuple
        self.deps_local_pred = None  # legacy: callable(did)->bool (mapped to table deps)
        self.deps_table_pred = None  # callable(did)->bool
        self.deps_owner_pred = None  # callable(did)->bool
        self.deps_table_func = None  # callable(zid)->List[int]
        self.deps_owner_func = None  # callable(zid)->List[int]
        self._locked_donors: Dict[int, List[int]] = {}

        self.send_queue: List[int] = []
        self.outbox: Dict[int, Dict[int, List[int]]] = {}  # dest_zid -> step_id -> atom ids

        self.diag: Dict[str, Any] = {
            "viol_w_gt1": 0,
            "viol_send_overlap": 0,
            "viol_lag": 0,
            "migrations": 0,
            "outbox_atoms": 0,
            "outbox_groups": 0,
            "table_rebuild_age": 0,
            "viol_buffer": 0,
            "wait_table": 0,
            "progress_epoch": 0,
            "wfg_samples": 0,
            "wfg_cycles": 0,
            "wfg_max_outdeg": 0,
            "wait_owner": 0,
            "wait_owner_unknown": 0,
            "wait_owner_stale": 0,
            "send_batches": 0,
            "send_batch_zones_total": 0,
            "send_batch_size_max": 0,
        }

    def _assert_invariants(self):
        if not self.formal_core:
            return
        w = [z.zid for z in self.zones if z.ztype == ZoneType.W]
        if len(w) > 1:
            self.diag["viol_w_gt1"] += 1
            raise RuntimeError(f"Formal core violated: >1 zones in W: {w}")

    def zone_id_for_z(self, zpos: float) -> int:
        zz = float(zpos) % self.box
        zid = int(zz / self.zwidth)
        if zid < 0:
            zid = 0
        if zid >= len(self.zones):
            zid = len(self.zones) - 1
        return zid

    def migrate_atoms_by_position(self, r: np.ndarray, source_zid: int, dest_step_id: int):
        src = self.zones[source_zid]
        if src.atom_ids.size == 0:
            return

        keep_src: List[int] = []
        for aid in src.atom_ids.tolist():
            dest = self.zone_id_for_z(float(r[int(aid), 2]))
            if dest == source_zid:
                keep_src.append(int(aid))
                continue
            dz = self.zones[dest]
            if dz.ztype == ZoneType.F:
                self.outbox.setdefault(dest, {}).setdefault(int(dest_step_id), []).append(int(aid))
            else:
                dz.atom_ids = (
                    np.array([int(aid)], dtype=np.int32)
                    if dz.atom_ids.size == 0
                    else np.concatenate([dz.atom_ids, np.array([int(aid)], np.int32)])
                )
                dz.table = None
            # remove from src
        src.atom_ids = np.array(keep_src, dtype=np.int32) if keep_src else np.empty((0,), np.int32)
        src.table = None
        self.diag["migrations"] += 1
        self.diag["outbox_atoms"] = sum(
            len(lst) for d in self.outbox.values() for lst in d.values()
        )
        self.diag["outbox_groups"] = sum(len(d) for d in self.outbox.values())

    def on_recv(self, zid: int, atom_ids: np.ndarray, step_id: int):
        z = self.zones[zid]
        z.atom_ids = atom_ids.astype(np.int32)
        z.halo_ids = np.empty((0,), np.int32)
        z.halo_step_id = -1
        z.step_id = int(step_id)
        z.table = None
        z.ztype = ZoneType.D if z.atom_ids.size else ZoneType.F
        self._assert_invariants()

    def _deps(self, zid: int) -> Tuple[List[int], float, float]:
        z = self.zones[zid]
        z0p = z.z0 - self.cutoff
        z1p = z.z1 + self.cutoff
        deps = zones_overlapping_range_pbc(z0p, z1p, self.box, self.zones)  # type: ignore[arg-type]
        return deps, z0p, z1p

    def set_deps_local_pred(self, pred):
        # legacy: treat as table-deps predicate
        self.deps_local_pred = pred
        self.deps_table_pred = pred

    def set_deps_preds(self, table_pred=None, owner_pred=None):
        self.deps_table_pred = table_pred
        self.deps_owner_pred = owner_pred

    def set_deps_funcs(self, table_func=None, owner_func=None):
        self.deps_table_func = table_func
        self.deps_owner_func = owner_func

    def set_priority_key(self, keyfunc):
        """Set total-order priority key for selecting next computable zone.
        keyfunc(zid)->tuple comparable lexicographically.
        """
        self.priority_key = keyfunc

    def _iter_zids(self):
        if self.priority_key is None:
            return list(self.order)
        try:
            return sorted(list(self.order), key=lambda zid: self.priority_key(int(zid)))
        except (TypeError, ValueError):
            return list(self.order)

    def _deps_table_missing(self, zid: int) -> list[int]:
        if self.deps_table_func is None or self.deps_table_pred is None:
            return []
        missing: list[int] = []
        for did in self.deps_table_func(int(zid)):
            if not bool(self.deps_table_pred(int(did))):
                missing.append(int(did))
        return missing

    def _deps_owner_missing(self, zid: int) -> list[int]:
        if self.deps_owner_func is None or self.deps_owner_pred is None:
            return []
        missing: list[int] = []
        for did in self.deps_owner_func(int(zid)):
            if not bool(self.deps_owner_pred(int(did))):
                missing.append(int(did))
        return missing

    def wfg_local_sample(self, *, max_edges: int = 256) -> dict[int, list[int]]:
        """Rank-local approximation of wait-for graph.

        Edges z->d mean: zone z is currently blocked waiting for donor zone d.
        No global synchronization is required.
        """
        g: dict[int, list[int]] = {}
        n_edges = 0
        for zid in self._iter_zids():
            if n_edges >= max_edges:
                break
            z = self.zones[int(zid)]
            if z.ztype == ZoneType.W:
                continue
            deps: list[int] = []
            deps.extend(self._deps_table_missing(int(zid)))
            deps.extend(self._deps_owner_missing(int(zid)))
            if deps:
                take = deps[: max(0, max_edges - n_edges)]
                g[int(zid)] = take
                n_edges += len(take)
        return g

    def wfg_local_cycle(self) -> list[int] | None:
        """Detect a cycle in local WFG sample (if any)."""
        from .graph_utils import find_cycle

        g = self.wfg_local_sample()
        return find_cycle(g)

    def record_wfg_sample(self) -> None:
        """Record WFG sample stats into diag counters."""
        g = self.wfg_local_sample()
        self.diag["wfg_samples"] = self.diag.get("wfg_samples", 0) + 1
        outdeg = [len(v) for v in g.values()] or [0]
        self.diag["wfg_max_outdeg"] = max(self.diag.get("wfg_max_outdeg", 0), max(outdeg))
        cyc = self.wfg_local_cycle()
        if cyc:
            self.diag["wfg_cycles"] = self.diag.get("wfg_cycles", 0) + 1
            self.diag["wfg_last_cycle"] = list(map(int, cyc))

    def _deps_table_ok(self, deps: List[int]) -> bool:
        pred = self.deps_table_pred
        if pred is None:
            return True
        for did in deps:
            dz = self.zones[did]
            if dz.ztype == ZoneType.F:
                continue
            if not bool(pred(int(did))):
                return False
        return True

    def _deps_owner_ok(self, deps: List[int]) -> bool:
        pred = self.deps_owner_pred
        if pred is None:
            return True
        for did in deps:
            dz = self.zones[did]
            if dz.ztype == ZoneType.F:
                continue
            if not bool(pred(int(did))):
                return False
        return True

    def _lag_ok(self, zid: int, deps: List[int]) -> bool:
        """Проверка time-lag между zone zid и её зависимостями.

        Для прототипа: требуем, чтобы зависимости не отставали больше max_step_lag.
        Если какой-то донор старее на >max_step_lag — старт вычисления запрещён.
        """
        if self.max_step_lag <= 0:
            return True
        zt = self.zones[zid].step_id
        for did in deps:
            dz = self.zones[did]
            if dz.ztype == ZoneType.F:
                continue
            if (zt - dz.step_id) > self.max_step_lag:
                return False
        return True

    def ensure_table(
        self, zid: int, r: np.ndarray, rc: float, skin_global: float, step: int, verlet_k_steps: int
    ):
        geom_aabb = None
        if self.geom_provider is not None:
            try:
                geom_aabb = self.geom_provider.geom(int(zid))
            except (KeyError, IndexError, AttributeError):
                geom_aabb = None
        z = self.zones[zid]
        deps, z0p, z1p = self._deps(zid)
        if self.deps_table_func is not None:
            deps = list(self.deps_table_func(int(zid)))

        # age rule (zone-time)
        if z.table is not None and self.table_max_age > 0:
            age = z.step_id - int(z.table.build_step)
            if age > self.table_max_age:
                z.table = None
                self.diag["table_rebuild_age"] += 1

        cand = []
        # Always include local atoms/halo explicitly (deps_table may exclude self)
        if z.atom_ids.size:
            cand.append(z.atom_ids)
        if z.halo_ids.size and z.halo_step_id == z.step_id:
            cand.append(z.halo_ids)
        for did in deps:
            if int(did) == int(zid):
                continue
            dz = self.zones[did]
            if dz.ztype != ZoneType.F:
                if dz.atom_ids.size:
                    cand.append(dz.atom_ids)
                if dz.halo_ids.size and dz.halo_step_id == dz.step_id:
                    cand.append(dz.halo_ids)
        candidate_ids = np.concatenate(cand).astype(np.int32) if cand else np.empty((0,), np.int32)
        # v4.2: 3D table implementation (CellList) if geom_aabb is provided
        if geom_aabb is not None:
            # candidate_ids already defines the support; build cell list on these ids
            cell = build_cell_list(r, candidate_ids.astype(np.int32), self.box, rc)
            # v4.3: store 3D AABB support geometry in table state
            mask_ok = mask_in_aabb_pbc(
                r, candidate_ids.astype(np.int32), geom_aabb.lo, geom_aabb.hi, rc, self.box
            )
            if not bool(mask_ok.all()):
                self.diag['tG3'] = self.diag.get('tG3', 0) + int(
                    (~mask_ok).sum()
                )  # table-geom violations (candidate outside support)
            z.table = InteractionTableState(
                impl=cell,
                candidate_ids=candidate_ids.astype(np.int32),
                rc=float(rc),
                build_step=int(step),
                z0p=float(z.z0 - rc),
                z1p=float(z.z1 + rc),
                lo=np.asarray(geom_aabb.lo, dtype=float),
                hi=np.asarray(geom_aabb.hi, dtype=float),
            )
            return
        if candidate_ids.size == 0:
            z.table = None
            return None

        impl = self.cache.get(
            zid,
            r,
            self.box,
            candidate_ids,
            rc=float(rc),
            skin_global=float(skin_global),
            step=int(step),
            verlet_k_steps=int(verlet_k_steps),
            z0=float(z0p),
            z1=float(z1p),
        )
        z.table = InteractionTableState(
            impl=impl,
            candidate_ids=candidate_ids,
            rc=float(rc),
            build_step=int(z.step_id),  # note: zone-time
            z0p=float(z0p),
            z1p=float(z1p),
        )
        return z.table

    def table_support(self, zid: int) -> np.ndarray:
        z = self.zones[zid]
        return z.table.support_ids() if z.table is not None else np.empty((0,), np.int32)

    def can_start_compute(self) -> bool:
        if self.work_zid is not None:
            return False
        for zid in self._iter_zids():
            z = self.zones[zid]
            if z.ztype == ZoneType.D and z.atom_ids.size:
                deps, _, _ = self._deps(zid)
                deps_table = deps
                deps_owner: List[int] = []
                if self.deps_owner_func is not None:
                    deps_owner = list(self.deps_owner_func(int(zid)))
                if self.deps_table_func is not None:
                    deps_table = list(self.deps_table_func(int(zid)))
                if (
                    self._deps_table_ok(deps_table)
                    and self._deps_owner_ok(deps_owner)
                    and self._lag_ok(zid, deps_table)
                ):
                    return True
        return False

    def start_compute(
        self, r: np.ndarray, rc: float, skin_global: float, step: int, verlet_k_steps: int
    ):
        self._assert_invariants()
        if self.work_zid is not None and self.formal_core:
            raise RuntimeError("Attempt to start compute while a zone is already in W")

        for zid in self._iter_zids():
            z = self.zones[zid]
            if z.ztype == ZoneType.D and z.atom_ids.size:
                deps, _, _ = self._deps(zid)
                deps_table = deps
                deps_owner: List[int] = []
                if self.deps_owner_func is not None:
                    deps_owner = list(self.deps_owner_func(int(zid)))
                if self.deps_table_func is not None:
                    deps_table = list(self.deps_table_func(int(zid)))
                table_ok = self._deps_table_ok(deps_table)
                owner_ok = self._deps_owner_ok(deps_owner)
                lag_ok = self._lag_ok(zid, deps_table)
                if not table_ok:
                    self.diag["wait_table"] += 1
                    continue
                if not owner_ok:
                    self.diag["wait_owner"] += 1
                    continue
                if not lag_ok:
                    self.diag["viol_lag"] += 1
                    continue

                donors = []
                for did in deps_table:
                    if did == zid:
                        continue
                    dz = self.zones[did]
                    if dz.ztype == ZoneType.D:
                        dz.ztype = ZoneType.P
                        donors.append(did)
                self._locked_donors[zid] = donors

                z.ztype = ZoneType.W
                self.work_zid = zid
                self.ensure_table(
                    zid,
                    r=r,
                    rc=rc,
                    skin_global=skin_global,
                    step=step,
                    verlet_k_steps=verlet_k_steps,
                )
                # v4.3: 3D halo geometric invariant check (I4) against table AABB support
                if z.table is not None and z.table.lo is not None and z.halo_ids.size > 0:
                    m = mask_in_aabb_pbc(
                        r,
                        z.halo_ids.astype(np.int32),
                        z.table.lo,
                        z.table.hi,
                        float(z.table.rc),
                        self.box,
                    )
                    if not bool(m.all()):
                        self.diag['hG3'] = self.diag.get('hG3', 0) + int((~m).sum())
                # v4.3: 3D halo-in-support invariant (I3) against table.support_ids
                if z.table is not None and z.halo_ids.size > 0:
                    sup = set(map(int, z.table.support_ids().tolist()))
                    bad = [int(a) for a in z.halo_ids.tolist() if int(a) not in sup]
                    if bad:
                        self.diag['hV3'] = self.diag.get('hV3', 0) + len(bad)
                self._assert_invariants()
                return zid
        return None

    def compute_step_for_work_zone(
        self,
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
        atom_types: np.ndarray | None = None,
        enable_step_id: bool = True,
    ):
        self._assert_invariants()
        if self.work_zid is None:
            return None
        zid = self.work_zid
        z = self.zones[zid]
        if z.ztype != ZoneType.W:
            raise RuntimeError("Internal: work_zid set but zone not in W")

        if z.atom_ids.size == 0:
            z.ztype = ZoneType.S
            self.send_queue.append(zid)
            self.work_zid = None
            return zid

        if z.table is None:
            self.ensure_table(
                zid, r=r, rc=rc, skin_global=skin_global, step=step, verlet_k_steps=verlet_k_steps
            )

        if atom_types is None:
            atom_types = np.ones((r.shape[0],), dtype=np.int32)
        else:
            atom_types = np.asarray(atom_types, dtype=np.int32)
        if hasattr(potential, "forces_on_targets"):
            support = z.table.support_ids() if z.table is not None else z.atom_ids
            if support.size == 0:
                support = z.atom_ids
            try:
                f = potential.forces_on_targets(
                    r=r,
                    box=self.box,
                    cutoff=cutoff,
                    rc=float(rc),
                    atom_types=atom_types,
                    target_ids=z.atom_ids,
                    candidate_ids=support,
                )
            except TypeError:
                # Backward compatibility: legacy force callbacks may not accept rc.
                f = potential.forces_on_targets(
                    r=r,
                    box=self.box,
                    cutoff=cutoff,
                    atom_types=atom_types,
                    target_ids=z.atom_ids,
                    candidate_ids=support,
                )
        elif z.table is not None and isinstance(z.table.impl, CellList):
            f = forces_on_targets_celllist_compact(
                r, self.box, potential, cutoff, z.atom_ids, z.table.impl, atom_types=atom_types
            )
        else:
            f = forces_on_targets_zonecells(
                r,
                self.box,
                potential,
                cutoff,
                z.atom_ids,
                z.table.impl if z.table else None,
                atom_types=atom_types,
            )
        vv_update_positions(r, v, mass, dt, self.box, z.atom_ids, f)

        dest_sid = z.step_id + (1 if enable_step_id else 0)
        self.migrate_atoms_by_position(r, zid, dest_step_id=dest_sid)

        self.ensure_table(
            zid, r=r, rc=rc, skin_global=skin_global, step=step, verlet_k_steps=verlet_k_steps
        )
        if z.table is not None and z.atom_ids.size:
            if hasattr(potential, "forces_on_targets"):
                support = z.table.support_ids() if z.table is not None else z.atom_ids
                if support.size == 0:
                    support = z.atom_ids
                try:
                    f2 = potential.forces_on_targets(
                        r=r,
                        box=self.box,
                        cutoff=cutoff,
                        rc=float(rc),
                        atom_types=atom_types,
                        target_ids=z.atom_ids,
                        candidate_ids=support,
                    )
                except TypeError:
                    # Backward compatibility: legacy force callbacks may not accept rc.
                    f2 = potential.forces_on_targets(
                        r=r,
                        box=self.box,
                        cutoff=cutoff,
                        atom_types=atom_types,
                        target_ids=z.atom_ids,
                        candidate_ids=support,
                    )
            elif z.table is not None and isinstance(z.table.impl, CellList):
                f2 = forces_on_targets_celllist_compact(
                    r, self.box, potential, cutoff, z.atom_ids, z.table.impl, atom_types=atom_types
                )
            else:
                f2 = forces_on_targets_zonecells(
                    r,
                    self.box,
                    potential,
                    cutoff,
                    z.atom_ids,
                    z.table.impl if z.table else None,
                    atom_types=atom_types,
                )
            vv_finish_velocities(v, mass, dt, z.atom_ids, f2)

        z.ztype = ZoneType.S
        if enable_step_id:
            z.step_id = dest_sid
        self.diag["progress_epoch"] = self.diag.get("progress_epoch", 0) + 1
        # ghosts are time-layer specific
        z.halo_ids = np.empty((0,), np.int32)
        z.halo_step_id = -1
        self.send_queue.append(zid)
        self.work_zid = None

        for did in self._locked_donors.get(zid, []):
            dz = self.zones[did]
            if dz.ztype == ZoneType.P:
                dz.ztype = ZoneType.D
        self._locked_donors.pop(zid, None)

        self._assert_invariants()
        return zid

    def pop_send_batch(self, batch_size: int) -> List[int]:
        if batch_size <= 0:
            batch_size = 1
        batch = self.send_queue[:batch_size]
        self.send_queue = self.send_queue[batch_size:]
        if batch:
            bs = int(len(batch))
            self.diag["send_batches"] = int(self.diag.get("send_batches", 0)) + 1
            self.diag["send_batch_zones_total"] = (
                int(self.diag.get("send_batch_zones_total", 0)) + bs
            )
            self.diag["send_batch_size_max"] = max(int(self.diag.get("send_batch_size_max", 0)), bs)
        return batch

    def iter_outbox_records(self):
        """Yield (dest_zid, step_id, ids_array) for all pending outbox groups."""
        for dest_zid, by_step in self.outbox.items():
            for sid, ids in by_step.items():
                arr = np.array(ids, dtype=np.int32) if ids else np.empty((0,), np.int32)
                yield int(dest_zid), int(sid), arr

    def clear_outbox(self):
        self.outbox.clear()
        self.diag["outbox_atoms"] = 0
        self.diag["outbox_groups"] = 0
