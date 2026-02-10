from __future__ import annotations

import numpy as np

from .state import minimum_image
from .zone_bins_localz import neighbor_candidates_for_atom


def forces_on_targets_zonecells(
    r: np.ndarray,
    box: float,
    potential,
    cutoff: float,
    target_ids: np.ndarray,
    zc,
    atom_types: np.ndarray | None = None,
) -> np.ndarray:
    cutoff2 = float(cutoff * cutoff)
    f = np.zeros((target_ids.size, 3), dtype=np.float64)
    for ti, i in enumerate(target_ids.tolist()):
        js = neighbor_candidates_for_atom(r[int(i)], box, zc)
        if js.size == 0:
            continue
        dr = r[int(i)][None, :] - r[js]
        dr = minimum_image(dr, box)
        r2 = (dr * dr).sum(axis=1)
        if atom_types is None:
            coef, _U = potential.pair(r2, cutoff2)
        else:
            ti_types = np.full(js.shape, int(atom_types[int(i)]), dtype=np.int32)
            tj_types = np.asarray(atom_types[js], dtype=np.int32)
            coef, _U = potential.pair(r2, cutoff2, type_i=ti_types, type_j=tj_types)
        f[ti] += (coef[:, None] * dr).sum(axis=0)
    return f


def forces_on_targets_celllist(
    r: np.ndarray,
    box: float,
    potential,
    cutoff: float,
    target_ids: np.ndarray,
    cell,
    atom_types: np.ndarray | None = None,
) -> np.ndarray:
    """Forces on `target_ids` using a global 3D CellList `cell`.

    This is a correctness-first kernel used for 3D zone decomposition bring-up.
    Complexity is O(N_target * n_neighbors) with cell-based candidate pruning.
    """
    from .state import minimum_image

    cutoff2 = float(cutoff * cutoff)
    f = np.zeros_like(r)
    # cell provides neighbor cell indices via its own helpers; keep generic:
    # We use cell.cell_atoms mapping and cell.ncell / rc.
    ncell = int(cell.ncell)
    # Precompute cell coordinates for each atom (already in cell if built consistently, but cheap):
    rr = r % box
    ic = np.floor(rr / cell.rc).astype(np.int32) % ncell

    # neighbor cell offsets (27)
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for i in target_ids.astype(np.int32):
        ci = tuple(int(x) for x in ic[i])
        fi = np.zeros(3, dtype=np.float64)
        ri = r[i]
        for dx, dy, dz in offsets:
            cj = ((ci[0] + dx) % ncell, (ci[1] + dy) % ncell, (ci[2] + dz) % ncell)
            ids = cell.cell_atoms.get(cj, None)
            if ids is None:
                continue
            # vectorized over candidates in that cell
            rj = r[ids]
            dr = minimum_image(ri - rj, box)
            r2 = np.sum(dr * dr, axis=1)
            if atom_types is None:
                coef, _U = potential.pair(r2, cutoff2)
            else:
                ti_types = np.full(ids.shape, int(atom_types[int(i)]), dtype=np.int32)
                tj_types = np.asarray(atom_types[ids], dtype=np.int32)
                coef, _U = potential.pair(r2, cutoff2, type_i=ti_types, type_j=tj_types)
            # force on i is -sum_j coef * dr_ij, but coef here is dU/dr / r
            fi += np.sum(coef[:, None] * dr, axis=0)
        f[i] = fi
    return f


def forces_on_targets_celllist_compact(
    r: np.ndarray,
    box: float,
    potential,
    cutoff: float,
    target_ids: np.ndarray,
    cell,
    atom_types: np.ndarray | None = None,
) -> np.ndarray:
    """Forces on target_ids using CellList; returns array (len(target_ids),3)."""
    f_full = forces_on_targets_celllist(
        r, box, potential, cutoff, target_ids, cell, atom_types=atom_types
    )
    return f_full[target_ids.astype(np.int32)]
