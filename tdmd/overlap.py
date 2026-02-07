from __future__ import annotations
import numpy as np
from .celllist import build_cell_list
from .state import minimum_image

def overlap_atoms_src_in_next_verlet(r: np.ndarray, box: float, src_ids: np.ndarray, next_ids: np.ndarray, rc: float) -> set:
    """Возвращает множество атомов из src_ids, которые попадают в rc-neighborhood атомов next_ids.

    Используется для правила: «атомы, входящие в таблицы взаимодействия следующей зоны, не передавать».
    """
    src_ids = np.asarray(src_ids, dtype=np.int32)
    next_ids = np.asarray(next_ids, dtype=np.int32)
    if src_ids.size == 0 or next_ids.size == 0:
        return set()
    cand = np.concatenate([src_ids, next_ids]).astype(np.int32)
    cl = build_cell_list(r, cand, box, rc=float(rc))
    rc2 = float(rc*rc)
    src_set = set(map(int, src_ids.tolist()))
    halo = set()
    for i in next_ids.tolist():
        ci = tuple(cl.idx[int(i)])
        neigh = []
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for dz in (-1,0,1):
                    cj = ((ci[0]+dx)%cl.ncell, (ci[1]+dy)%cl.ncell, (ci[2]+dz)%cl.ncell)
                    arr = cl.cell_atoms.get(cj)
                    if arr is not None and arr.size:
                        neigh.append(arr)
        if not neigh:
            continue
        js = np.concatenate(neigh)
        dr = r[int(i)][None,:] - r[js]
        dr = minimum_image(dr, box)
        r2 = (dr*dr).sum(axis=1)
        mask = (r2>0.0) & (r2<=rc2)
        for j, ok in zip(js.tolist(), mask.tolist()):
            if ok and int(j) in src_set:
                halo.add(int(j))
    return halo
