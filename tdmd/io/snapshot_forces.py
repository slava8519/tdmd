from __future__ import annotations

import numpy as np

from ..celllist import forces_on_targets_celllist


def compute_forces_snapshot(
    r: np.ndarray,
    box: float,
    potential,
    cutoff: float,
    atom_types: np.ndarray | None = None,
) -> np.ndarray:
    rr = np.asarray(r, dtype=float)
    if rr.ndim != 2 or rr.shape[1] != 3:
        raise ValueError("r must have shape (N,3)")
    n = int(rr.shape[0])
    ids = np.arange(n, dtype=np.int32)
    if hasattr(potential, "forces_energy_virial"):
        f, _pe, _w = potential.forces_energy_virial(
            rr,
            float(box),
            float(cutoff),
            atom_types if atom_types is None else np.asarray(atom_types, dtype=np.int32),
        )
        return np.asarray(f, dtype=float)
    return forces_on_targets_celllist(
        rr,
        float(box),
        potential,
        float(cutoff),
        ids,
        ids,
        rc=float(cutoff),
        atom_types=None if atom_types is None else np.asarray(atom_types, dtype=np.int32),
    )

