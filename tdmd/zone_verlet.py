from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .state import minimum_image


@dataclass
class ZoneVerlet:
    rc: float
    candidate_ids: np.ndarray  # int32
    neighbors: Dict[int, np.ndarray]  # atom id -> neighbor ids (within rc) among candidate_ids
    r_ref: np.ndarray  # positions snapshot for candidate_ids (same order)
    last_build_step: int


def _build_neighbors(
    r: np.ndarray, box: float, candidate_ids: np.ndarray, rc: float
) -> Dict[int, np.ndarray]:
    # cell list on candidate set
    rc = float(rc)
    ncell = max(1, int(box / rc))
    idx_all = np.floor((r % box) / rc).astype(int) % ncell

    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for i in candidate_ids.tolist():
        key = (int(idx_all[i, 0]), int(idx_all[i, 1]), int(idx_all[i, 2]))
        buckets.setdefault(key, []).append(int(i))

    rc2 = rc * rc
    neigh: Dict[int, List[int]] = {int(i): [] for i in candidate_ids.tolist()}

    for i in candidate_ids.tolist():
        ci = tuple(idx_all[i])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cj = ((ci[0] + dx) % ncell, (ci[1] + dy) % ncell, (ci[2] + dz) % ncell)
                    js = buckets.get(cj, [])
                    if not js:
                        continue
                    dr = r[i][None, :] - r[js]
                    dr = minimum_image(dr, box)
                    r2 = (dr * dr).sum(axis=1)
                    for j, ok in zip(js, (r2 > 0.0) & (r2 <= rc2)):
                        if ok:
                            neigh[int(i)].append(int(j))

    return {k: np.array(v, dtype=np.int32) for k, v in neigh.items()}


def _max_disp(r: np.ndarray, box: float, candidate_ids: np.ndarray, r_ref: np.ndarray) -> float:
    rr = r[candidate_ids]
    dr = rr - r_ref
    dr = minimum_image(dr, box)
    disp = np.sqrt((dr * dr).sum(axis=1))
    return float(disp.max()) if disp.size else 0.0


class ZoneVerletCache:
    """Кэш зонных Verlet-таблиц на одном MPI ранге."""

    def __init__(self):
        self._cache: Dict[int, ZoneVerlet] = {}

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
    ) -> ZoneVerlet:
        candidate_ids = np.asarray(candidate_ids, dtype=np.int32)
        zv = self._cache.get(int(zid))
        need = False
        if zv is None:
            need = True
        else:
            if float(zv.rc) != float(rc):
                need = True
            else:
                # rebuild on schedule
                if (step - zv.last_build_step) >= max(1, int(verlet_k_steps)):
                    need = True
                else:
                    # if candidate set changed, rebuild (simplest safe rule)
                    if candidate_ids.size != zv.candidate_ids.size or not np.array_equal(
                        candidate_ids, zv.candidate_ids
                    ):
                        need = True
                    else:
                        if skin_global > 0.0:
                            if _max_disp(r, box, candidate_ids, zv.r_ref) > 0.5 * skin_global:
                                need = True

        if need:
            neighbors = _build_neighbors(r, box, candidate_ids, rc)
            r_ref = r[candidate_ids].copy()
            zv = ZoneVerlet(
                rc=float(rc),
                candidate_ids=candidate_ids.copy(),
                neighbors=neighbors,
                r_ref=r_ref,
                last_build_step=int(step),
            )
            self._cache[int(zid)] = zv
        return zv
