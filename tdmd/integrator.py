from __future__ import annotations

import numpy as np

from .state import pbc_wrap


def vv_update_positions(r, v, mass, dt, box, ids, f):
    if np.isscalar(mass):
        m = float(mass)
        if m <= 0.0:
            raise ValueError("mass must be positive")
        inv_m = 1.0 / m
        v_half = v[ids] + 0.5 * dt * f * inv_m
    else:
        masses = np.asarray(mass, dtype=float)
        if masses.ndim != 1:
            raise ValueError("mass array must be 1D")
        if np.any(masses <= 0.0):
            raise ValueError("all masses must be positive")
        v_half = v[ids] + 0.5 * dt * f / masses[ids][:, None]
    r_new = pbc_wrap(r[ids] + dt * v_half, box)
    r[ids] = r_new
    v[ids] = v_half


def vv_finish_velocities(v, mass, dt, ids, f_new):
    if np.isscalar(mass):
        m = float(mass)
        if m <= 0.0:
            raise ValueError("mass must be positive")
        inv_m = 1.0 / m
        v[ids] = v[ids] + 0.5 * dt * f_new * inv_m
    else:
        masses = np.asarray(mass, dtype=float)
        if masses.ndim != 1:
            raise ValueError("mass array must be 1D")
        if np.any(masses <= 0.0):
            raise ValueError("all masses must be positive")
        v[ids] = v[ids] + 0.5 * dt * f_new / masses[ids][:, None]
