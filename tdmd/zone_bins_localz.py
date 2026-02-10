from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .constants import GEOM_EPSILON
from .state import minimum_image


@dataclass
class ZoneUnwrappedZBins:
    rc: float
    ncell_xy: int
    ncell_z: int
    origin_z: float
    width_z: float
    wraps: bool
    cell_atoms: Dict[Tuple[int, int, int], np.ndarray]  # (ix,iy,iz) -> atom ids
    candidate_ids: np.ndarray
    r_ref: np.ndarray
    last_build_step: int


def _interval_unwrapped(z0: float, z1: float, box: float):
    """Параметры p-интервала по z в развёрнутом представлении."""
    width = float(z1 - z0)
    if width <= 0 or box <= 0:
        return 0.0, float(box), False
    origin = float(z0 % box)
    wraps = (origin + width) > box
    return origin, width, wraps


def _unwrapped_z(z_mod: float, box: float, origin_z: float, wraps: bool) -> float:
    if wraps and z_mod < origin_z:
        return z_mod + box
    return z_mod


def _max_disp(r: np.ndarray, box: float, candidate_ids: np.ndarray, r_ref: np.ndarray) -> float:
    if candidate_ids.size == 0:
        return 0.0
    rr = r[candidate_ids]
    dr = rr - r_ref
    dr = minimum_image(dr, box)
    disp = np.sqrt((dr * dr).sum(axis=1))
    return float(disp.max()) if disp.size else 0.0


def _cell_index_xy(pos: np.ndarray, box: float, rc: float, ncell_xy: int):
    x = pos[0] % box
    y = pos[1] % box
    ix = int(np.floor(x / rc)) % ncell_xy
    iy = int(np.floor(y / rc)) % ncell_xy
    return ix, iy


def _cell_index_z(
    pos: np.ndarray, box: float, rc: float, origin_z: float, ncell_z: int, wraps: bool
):
    z = float(pos[2] % box)
    zu = _unwrapped_z(z, box, origin_z, wraps)
    zz = zu - origin_z
    if zz < 0.0:
        zz = 0.0
    iz = int(np.floor(zz / rc))
    if iz < 0:
        iz = 0
    if iz >= ncell_z:
        iz = ncell_z - 1
    return iz


def _build_bins(
    r: np.ndarray, box: float, candidate_ids: np.ndarray, rc: float, z0: float, z1: float, step: int
) -> ZoneUnwrappedZBins:
    rc = float(rc)
    ncell_xy = max(1, int(box / rc))
    origin_z, width_z, wraps = _interval_unwrapped(z0, z1, box)
    ncell_z = max(1, int(np.ceil(width_z / rc)))

    candidate_ids = np.asarray(candidate_ids, dtype=np.int32)
    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for i in candidate_ids.tolist():
        pos = r[int(i)]
        ix, iy = _cell_index_xy(pos, box, rc, ncell_xy)
        iz = _cell_index_z(pos, box, rc, origin_z, ncell_z, wraps)
        buckets.setdefault((ix, iy, iz), []).append(int(i))
    cell_atoms = {k: np.array(v, dtype=np.int32) for k, v in buckets.items()}

    return ZoneUnwrappedZBins(
        rc=rc,
        ncell_xy=ncell_xy,
        ncell_z=ncell_z,
        origin_z=origin_z,
        width_z=width_z,
        wraps=wraps,
        cell_atoms=cell_atoms,
        candidate_ids=candidate_ids.copy(),
        r_ref=r[candidate_ids].copy(),
        last_build_step=int(step),
    )


class PersistentZoneLocalZBinsCache:
    def __init__(self):
        self._cache: Dict[int, ZoneUnwrappedZBins] = {}

    def get(
        self,
        zid: int,
        r: np.ndarray,
        box: float,
        candidate_ids: np.ndarray,
        rc: float,
        skin_global: float,
        step: int,
        verlet_k_steps: int,
        z0: float,
        z1: float,
    ) -> ZoneUnwrappedZBins:
        zid = int(zid)
        candidate_ids = np.asarray(candidate_ids, dtype=np.int32)
        origin_z, width_z, wraps = _interval_unwrapped(z0, z1, box)

        zc = self._cache.get(zid)
        need_rebuild = False
        if zc is None:
            need_rebuild = True
        else:
            if float(zc.rc) != float(rc):
                need_rebuild = True
            elif (
                abs(zc.origin_z - origin_z) > GEOM_EPSILON
                or abs(zc.width_z - width_z) > 1e-9
                or bool(zc.wraps) != bool(wraps)
            ):
                need_rebuild = True
            else:
                if (step - zc.last_build_step) >= max(1, int(verlet_k_steps)):
                    need_rebuild = True
                else:
                    if skin_global > 0.0 and zc.r_ref.shape[0] == candidate_ids.shape[0]:
                        if _max_disp(r, box, candidate_ids, zc.r_ref) > 0.5 * skin_global:
                            need_rebuild = True

        if need_rebuild:
            zc = _build_bins(r, box, candidate_ids, rc=rc, z0=z0, z1=z1, step=step)
            self._cache[zid] = zc
        else:
            buckets: Dict[Tuple[int, int, int], List[int]] = {}
            for i in candidate_ids.tolist():
                pos = r[int(i)]
                ix, iy = _cell_index_xy(pos, box, float(rc), zc.ncell_xy)
                iz = _cell_index_z(pos, box, float(rc), zc.origin_z, zc.ncell_z, zc.wraps)
                buckets.setdefault((ix, iy, iz), []).append(int(i))
            zc.cell_atoms = {k: np.array(v, dtype=np.int32) for k, v in buckets.items()}
            zc.candidate_ids = candidate_ids
        return zc


def neighbor_candidates_for_atom(r_i: np.ndarray, box: float, zc: ZoneUnwrappedZBins):
    rc = float(zc.rc)
    ix = int(np.floor((r_i[0] % box) / rc)) % zc.ncell_xy
    iy = int(np.floor((r_i[1] % box) / rc)) % zc.ncell_xy

    zmod = float(r_i[2] % box)
    zu = _unwrapped_z(zmod, box, zc.origin_z, zc.wraps)
    zz = zu - zc.origin_z
    if zz < 0.0:
        zz = 0.0
    iz = int(np.floor(zz / rc))
    if iz < 0:
        iz = 0
    if iz >= zc.ncell_z:
        iz = zc.ncell_z - 1

    neigh = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                kz = iz + dz
                if kz < 0 or kz >= zc.ncell_z:
                    continue
                cj = ((ix + dx) % zc.ncell_xy, (iy + dy) % zc.ncell_xy, kz)
                arr = zc.cell_atoms.get(cj)
                if arr is not None and arr.size:
                    neigh.append(arr)
    return np.concatenate(neigh) if neigh else np.empty((0,), np.int32)


# v1.6: dissertation-semantics support set
def support_ids(zc) -> np.ndarray:
    """Return atoms that *enter the interaction table* (support set).
    For our bins-based table, this is simply candidate_ids used to build bins.
    """
    return getattr(zc, 'candidate_ids', np.empty((0,), np.int32))
