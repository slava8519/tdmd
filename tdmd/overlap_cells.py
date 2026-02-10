from __future__ import annotations

import numpy as np

from .state import minimum_image
from .zone_bins_localz import neighbor_candidates_for_atom


def overlap_atoms_src_in_next_zonecells(
    r: np.ndarray, box: float, src_ids: np.ndarray, next_ids: np.ndarray, rc: float, next_zc
) -> set:
    src_ids = np.asarray(src_ids, dtype=np.int32)
    next_ids = np.asarray(next_ids, dtype=np.int32)
    if src_ids.size == 0 or next_ids.size == 0:
        return set()
    src_set = set(map(int, src_ids.tolist()))
    rc2 = float(rc * rc)
    halo = set()
    for i in next_ids.tolist():
        js = neighbor_candidates_for_atom(r[int(i)], box, next_zc)
        if js.size == 0:
            continue
        dr = r[int(i)][None, :] - r[js]
        dr = minimum_image(dr, box)
        r2 = (dr * dr).sum(axis=1)
        mask = (r2 > 0.0) & (r2 <= rc2)
        for j, ok in zip(js.tolist(), mask.tolist()):
            if ok and int(j) in src_set:
                halo.add(int(j))
    return halo
