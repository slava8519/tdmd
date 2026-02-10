from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .state import minimum_image


@dataclass
class CellList:
    rc: float
    ncell: int
    cell_atoms: Dict[Tuple[int, int, int], np.ndarray]
    idx: np.ndarray  # per atom cell index (N,3), for referenced atom subset mapping not needed


def build_cell_list(r: np.ndarray, ids: np.ndarray, box: float, rc: float) -> CellList:
    rc = float(rc)
    ncell = max(1, int(box / rc))
    # compute indices for all atoms (for simplicity), but we only populate cell_atoms for ids
    idx_all = np.floor((r % box) / rc).astype(int) % ncell
    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for i in ids.tolist():
        key = (int(idx_all[i, 0]), int(idx_all[i, 1]), int(idx_all[i, 2]))
        buckets.setdefault(key, []).append(int(i))
    cell_atoms = {k: np.array(v, dtype=np.int32) for k, v in buckets.items()}
    return CellList(rc=rc, ncell=ncell, cell_atoms=cell_atoms, idx=idx_all)


def forces_on_targets_celllist(
    r: np.ndarray,
    box: float,
    potential,
    cutoff: float,
    target_ids: np.ndarray,
    candidate_ids: np.ndarray,
    rc: float,
    atom_types: np.ndarray | None = None,
) -> np.ndarray:
    """Силы на target_ids, используя cell-list по candidate_ids.

    rc обычно = cutoff + skin_global (Verlet радиус), но в силе используем только cutoff.
    """
    cutoff2 = float(cutoff * cutoff)
    cl = build_cell_list(r, candidate_ids, box, rc=rc)
    f = np.zeros((target_ids.size, 3), dtype=np.float64)

    for ti, i in enumerate(target_ids.tolist()):
        ci = tuple(cl.idx[i])
        neigh_ids = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cj = ((ci[0] + dx) % cl.ncell, (ci[1] + dy) % cl.ncell, (ci[2] + dz) % cl.ncell)
                    arr = cl.cell_atoms.get(cj)
                    if arr is not None and arr.size:
                        neigh_ids.append(arr)
        if not neigh_ids:
            continue
        js = np.concatenate(neigh_ids)
        dr = r[i][None, :] - r[js]
        dr = minimum_image(dr, box)
        r2 = (dr * dr).sum(axis=1)
        if atom_types is None:
            coef, _U = potential.pair(r2, cutoff2)
        else:
            ti_types = np.full(js.shape, int(atom_types[i]), dtype=np.int32)
            tj_types = np.asarray(atom_types[js], dtype=np.int32)
            coef, _U = potential.pair(r2, cutoff2, type_i=ti_types, type_j=tj_types)
        f[ti] += (coef[:, None] * dr).sum(axis=0)

    return f
