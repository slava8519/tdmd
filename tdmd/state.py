from __future__ import annotations
from dataclasses import dataclass
from typing import Union
import numpy as np

@dataclass
class GlobalState:
    r: np.ndarray
    v: np.ndarray
    t: int = 0

def init_positions(n_atoms: int, box: float, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n_atoms,3)) * box

def init_velocities(n_atoms: int, temperature: float, mass: float, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed+1234)
    std = np.sqrt(temperature/mass)
    v = rng.normal(0.0,std,size=(n_atoms,3))
    v -= v.mean(axis=0, keepdims=True)
    return v

def pbc_wrap(r: np.ndarray, box: float) -> np.ndarray:
    return r - box * np.floor(r/box)

def minimum_image(dr: np.ndarray, box: float) -> np.ndarray:
    return dr - box * np.round(dr/box)

def kinetic_energy(v: np.ndarray, mass: Union[float, np.ndarray]) -> float:
    if np.isscalar(mass):
        m = float(mass)
        if m <= 0.0:
            raise ValueError("mass must be positive")
        return 0.5 * m * float((v * v).sum())
    masses = np.asarray(mass, dtype=float)
    if masses.ndim != 1 or masses.shape[0] != v.shape[0]:
        raise ValueError("mass array must have shape (N,)")
    if np.any(masses <= 0.0):
        raise ValueError("all masses must be positive")
    return 0.5 * float((masses[:, None] * (v * v)).sum())

def temperature_from_ke(ke: float, n_atoms: int) -> float:
    dof = max(1, 3*n_atoms-3)
    return 2.0*ke/dof
