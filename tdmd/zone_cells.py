from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .state import minimum_image


@dataclass
class ZoneCells:
    rc: float
    ncell: int
    cell_atoms: Dict[Tuple[int, int, int], np.ndarray]
    idx_all: np.ndarray  # (N,3) cell indices for all atoms
    candidate_ids: np.ndarray
    r_ref: np.ndarray  # (Nc,3) snapshot for candidates
    last_build_step: int


def _build_zone_cells(r: np.ndarray, box: float, candidate_ids: np.ndarray, rc: float) -> ZoneCells:
    rc = float(rc)
    ncell = max(1, int(box / rc))
    idx_all = np.floor((r % box) / rc).astype(int) % ncell
    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for i in candidate_ids.tolist():
        key = (int(idx_all[i, 0]), int(idx_all[i, 1]), int(idx_all[i, 2]))
        buckets.setdefault(key, []).append(int(i))
    cell_atoms = {k: np.array(v, dtype=np.int32) for k, v in buckets.items()}
    r_ref = r[candidate_ids].copy()
    return ZoneCells(
        rc=rc,
        ncell=ncell,
        cell_atoms=cell_atoms,
        idx_all=idx_all,
        candidate_ids=candidate_ids.copy(),
        r_ref=r_ref,
        last_build_step=0,
    )


def _max_disp(r: np.ndarray, box: float, candidate_ids: np.ndarray, r_ref: np.ndarray) -> float:
    rr = r[candidate_ids]
    dr = rr - r_ref
    dr = minimum_image(dr, box)
    disp = np.sqrt((dr * dr).sum(axis=1))
    return float(disp.max()) if disp.size else 0.0


class ZoneCellCache:
    def __init__(self):
        self._cache: Dict[int, ZoneCells] = {}

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
    ) -> ZoneCells:
        zid = int(zid)
        candidate_ids = np.asarray(candidate_ids, dtype=np.int32)
        zc = self._cache.get(zid)
        need = False
        if zc is None:
            need = True
        else:
            if float(zc.rc) != float(rc):
                need = True
            else:
                if (step - zc.last_build_step) >= max(1, int(verlet_k_steps)):
                    need = True
                else:
                    if candidate_ids.size != zc.candidate_ids.size or not np.array_equal(
                        candidate_ids, zc.candidate_ids
                    ):
                        need = True
                    else:
                        if (
                            skin_global > 0.0
                            and _max_disp(r, box, candidate_ids, zc.r_ref) > 0.5 * skin_global
                        ):
                            need = True
        if need:
            zc = _build_zone_cells(r, box, candidate_ids, rc=rc)
            zc.last_build_step = int(step)
            self._cache[zid] = zc
        return zc
