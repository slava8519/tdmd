from __future__ import annotations

from typing import Union

import numpy as np


def normalize_mass(
    mass: Union[float, np.ndarray], *, n_atoms: int
) -> tuple[float | None, np.ndarray | None, float | None]:
    """Normalize a mass input into either a scalar or a per-atom array.

    Returns:
      - mass_scalar: float if scalar mass was provided, else None
      - mass_array: (N,) float array if per-atom masses were provided, else None
      - inv_mass_scalar: 1/mass_scalar if scalar, else None
    """
    if isinstance(mass, np.ndarray):
        mass_arr = np.asarray(mass, dtype=float)
        if mass_arr.ndim != 1 or int(mass_arr.shape[0]) != int(n_atoms):
            raise ValueError("mass array must have shape (N,)")
        if np.any(mass_arr <= 0.0):
            raise ValueError("all masses must be positive")
        return None, mass_arr, None

    mass_scalar = float(mass)
    if mass_scalar <= 0.0:
        raise ValueError("mass must be positive")
    return mass_scalar, None, 1.0 / mass_scalar


def normalize_atom_types(atom_types: np.ndarray | None, *, n_atoms: int) -> np.ndarray:
    if atom_types is None:
        return np.ones(int(n_atoms), dtype=np.int32)
    out = np.asarray(atom_types, dtype=np.int32)
    if out.ndim != 1 or int(out.shape[0]) != int(n_atoms):
        raise ValueError("atom_types must have shape (N,)")
    return out
