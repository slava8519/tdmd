from __future__ import annotations
from typing import Union
import numpy as np
from .state import minimum_image

def compute_observables(
    r: np.ndarray,
    v: np.ndarray,
    mass: Union[float, np.ndarray],
    box: float,
    potential,
    cutoff: float,
    atom_types: np.ndarray | None = None,
) -> dict:
    """Compute basic thermodynamic observables (kB=1).

    Returns:
        dict with KE, PE, E, T, P, virial
    Notes:
        Pressure uses virial expression:
            P = N*T/V + W/(3V), W = sum_{i<j} r_ij · f_ij
    """
    N = int(r.shape[0])
    V = float(box**3)
    if np.isscalar(mass):
        m = float(mass)
        if m <= 0.0:
            raise ValueError("mass must be positive")
        ke = 0.5 * m * float((v * v).sum())
    else:
        masses = np.asarray(mass, dtype=float)
        if masses.ndim != 1 or masses.shape[0] != N:
            raise ValueError("mass array must have shape (N,)")
        if np.any(masses <= 0.0):
            raise ValueError("all masses must be positive")
        ke = 0.5 * float((masses[:, None] * (v * v)).sum())
    T = (2.0*ke)/(3.0*max(1, N))  # kB=1
    if atom_types is None:
        atom_types = np.ones((N,), dtype=np.int32)
    else:
        atom_types = np.asarray(atom_types, dtype=np.int32)
        if atom_types.ndim != 1 or atom_types.shape[0] != N:
            raise ValueError("atom_types must have shape (N,)")

    # Many-body potentials (EAM): use dedicated backend if available.
    if hasattr(potential, "energy_virial"):
        pe, W = potential.energy_virial(r, box, cutoff, atom_types)
    else:
        # Pairwise (vectorized) for tests: O(N^2) but N is small in verify bench.
        dr = r[:,None,:] - r[None,:,:]
        dr = minimum_image(dr, box)
        r2 = (dr*dr).sum(axis=2)
        cutoff2 = float(cutoff*cutoff)

        # upper triangle mask excluding diagonal
        iu = np.triu_indices(N, k=1)
        r2u = r2[iu]
        mask = (r2u > 0.0) & (r2u < cutoff2)
        if not np.any(mask):
            pe = 0.0
            W = 0.0
        else:
            r2m = r2u[mask]
            ti = atom_types[iu[0]][mask]
            tj = atom_types[iu[1]][mask]
            coef, U = potential.pair(r2m, cutoff2, type_i=ti, type_j=tj)
            pe = float(U.sum())
            # virial: sum r_ij · f_ij = coef * |r|^2
            W = float((coef * r2m).sum())

    P = (N*T)/V + W/(3.0*V)
    return {"N": N, "V": V, "KE": ke, "PE": pe, "E": ke+pe, "T": T, "P": P, "virial": W}
