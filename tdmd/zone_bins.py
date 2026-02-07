from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
from .state import minimum_image

@dataclass
class ZoneCellBins:
    rc: float
    ncell: int
    cell_atoms: Dict[Tuple[int,int,int], np.ndarray]
    idx_all: np.ndarray  # (N,3) indices for all atoms (recomputed on rebuild)
    candidate_ids: np.ndarray  # last candidate ids used to fill bins (not used for rebuild decisions)
    r_ref: np.ndarray  # snapshot for candidate_ids (same order)
    last_build_step: int

    def update_bins(self, r: np.ndarray, candidate_ids: np.ndarray, box: float):
        """Обновить содержимое ячеек без пересоздания объекта.

        Это дешёвая операция: очистить mapping и заново распределить candidate_ids по ячейкам.
        """
        candidate_ids = np.asarray(candidate_ids, dtype=np.int32)
        self.candidate_ids = candidate_ids
        # NOTE: idx_all is defined for all atoms; it can be updated cheaply too, but we keep it consistent on rebuild.
        buckets: Dict[Tuple[int,int,int], List[int]] = {}
        for i in candidate_ids.tolist():
            key = (int(self.idx_all[i,0]), int(self.idx_all[i,1]), int(self.idx_all[i,2]))
            buckets.setdefault(key, []).append(int(i))
        self.cell_atoms = {k: np.array(v, dtype=np.int32) for k,v in buckets.items()}

def _max_disp(r: np.ndarray, box: float, candidate_ids: np.ndarray, r_ref: np.ndarray) -> float:
    if candidate_ids.size == 0:
        return 0.0
    rr = r[candidate_ids]
    dr = rr - r_ref
    dr = minimum_image(dr, box)
    disp = np.sqrt((dr*dr).sum(axis=1))
    return float(disp.max()) if disp.size else 0.0

def _new_bins(r: np.ndarray, box: float, candidate_ids: np.ndarray, rc: float, step: int) -> ZoneCellBins:
    rc = float(rc)
    ncell = max(1, int(box / rc))
    idx_all = np.floor((r % box) / rc).astype(int) % ncell
    candidate_ids = np.asarray(candidate_ids, dtype=np.int32)
    bins = ZoneCellBins(
        rc=rc, ncell=ncell, cell_atoms={}, idx_all=idx_all,
        candidate_ids=candidate_ids.copy(), r_ref=r[candidate_ids].copy(), last_build_step=int(step)
    )
    bins.update_bins(r, candidate_ids, box)
    return bins

class PersistentZoneBinsCache:
    """Персистентный кэш ячейных корзин по зонам.

    Важное отличие от v1.1:
    - изменение состава candidate_ids НЕ триггерит rebuild структуры;
      вместо этого мы просто обновляем bins.update_bins(...).
    - rebuild происходит только по физически оправданным причинам (rc/смещение/период).
    """
    def __init__(self):
        self._cache: Dict[int, ZoneCellBins] = {}

    def get(self, zid: int, r: np.ndarray, box: float, candidate_ids: np.ndarray,
            rc: float, skin_global: float, step: int, verlet_k_steps: int) -> ZoneCellBins:
        zid = int(zid)
        candidate_ids = np.asarray(candidate_ids, dtype=np.int32)
        zc = self._cache.get(zid)
        need_rebuild = False
        if zc is None:
            need_rebuild = True
        else:
            if float(zc.rc) != float(rc):
                need_rebuild = True
            else:
                if (step - zc.last_build_step) >= max(1, int(verlet_k_steps)):
                    need_rebuild = True
                else:
                    if skin_global > 0.0 and zc.r_ref.shape[0] == candidate_ids.shape[0]:
                        # displacement check only if we can compare apples-to-apples;
                        # if size differs, we skip (rebuild will happen on schedule).
                        if _max_disp(r, box, candidate_ids, zc.r_ref) > 0.5*skin_global:
                            need_rebuild = True

        if need_rebuild:
            zc = _new_bins(r, box, candidate_ids, rc=rc, step=step)
            self._cache[zid] = zc
        else:
            # cheap update for new candidate_ids
            zc.update_bins(r, candidate_ids, box)
            # refresh snapshot occasionally to keep displacement meaningful
            if (step - zc.last_build_step) >= max(1, int(verlet_k_steps)):
                zc.r_ref = r[candidate_ids].copy()
                zc.last_build_step = int(step)

        return zc
