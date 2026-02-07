from __future__ import annotations
import numpy as np
from .state import minimum_image

def forces_on_targets_neighbor_dict(r: np.ndarray, box: float, potential, cutoff: float,
                                    target_ids: np.ndarray, neighbor_dict,
                                    atom_types: np.ndarray | None = None) -> np.ndarray:
    cutoff2 = float(cutoff*cutoff)
    f = np.zeros((target_ids.size, 3), dtype=np.float64)
    for ti, i in enumerate(target_ids.tolist()):
        js = neighbor_dict.get(int(i))
        if js is None or js.size == 0:
            continue
        dr = r[int(i)][None,:] - r[js]
        dr = minimum_image(dr, box)
        r2 = (dr*dr).sum(axis=1)
        if atom_types is None:
            coef, _U = potential.pair(r2, cutoff2)
        else:
            ti_types = np.full(js.shape, int(atom_types[int(i)]), dtype=np.int32)
            tj_types = np.asarray(atom_types[js], dtype=np.int32)
            coef, _U = potential.pair(r2, cutoff2, type_i=ti_types, type_j=tj_types)
        f[ti] += (coef[:,None]*dr).sum(axis=0)
    return f
