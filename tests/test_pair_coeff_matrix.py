from __future__ import annotations

import numpy as np

from tdmd.celllist import forces_on_targets_celllist
from tdmd.observables import compute_observables
from tdmd.potentials import make_potential
from tdmd.state import minimum_image


def _bruteforce_forces(
    r: np.ndarray, box: float, cutoff: float, potential, atom_types: np.ndarray
) -> np.ndarray:
    n = int(r.shape[0])
    out = np.zeros_like(r)
    cutoff2 = float(cutoff * cutoff)
    for i in range(n):
        fi = np.zeros(3, dtype=float)
        for j in range(n):
            if i == j:
                continue
            dr = minimum_image(r[i] - r[j], box)
            r2 = float(np.dot(dr, dr))
            coef, _ = potential.pair(
                np.array([r2], dtype=float),
                cutoff2,
                type_i=np.array([int(atom_types[i])], dtype=np.int32),
                type_j=np.array([int(atom_types[j])], dtype=np.int32),
            )
            fi += float(coef[0]) * dr
        out[i] = fi
    return out


def _bruteforce_pe(
    r: np.ndarray, box: float, cutoff: float, potential, atom_types: np.ndarray
) -> float:
    n = int(r.shape[0])
    cutoff2 = float(cutoff * cutoff)
    pe = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dr = minimum_image(r[i] - r[j], box)
            r2 = float(np.dot(dr, dr))
            _, u = potential.pair(
                np.array([r2], dtype=float),
                cutoff2,
                type_i=np.array([int(atom_types[i])], dtype=np.int32),
                type_j=np.array([int(atom_types[j])], dtype=np.int32),
            )
            pe += float(u[0])
    return pe


def test_lj_pair_coeff_matrix_matches_bruteforce():
    r = np.array(
        [
            [1.0, 1.1, 1.2],
            [2.0, 2.1, 1.8],
            [3.2, 2.7, 2.9],
            [4.0, 4.2, 3.6],
        ],
        dtype=float,
    )
    v = np.zeros_like(r)
    box = 12.0
    cutoff = 4.0
    atom_types = np.array([1, 2, 2, 1], dtype=np.int32)
    potential = make_potential(
        "lj",
        {
            "pair_coeffs": {
                "1-1": {"epsilon": 1.0, "sigma": 1.0},
                "1-2": {"epsilon": 0.6, "sigma": 0.9},
                "2-2": {"epsilon": 0.8, "sigma": 1.1},
            }
        },
    )

    ids = np.arange(r.shape[0], dtype=np.int32)
    f_cell = forces_on_targets_celllist(
        r, box, potential, cutoff, ids, ids, rc=cutoff, atom_types=atom_types
    )
    f_ref = _bruteforce_forces(r, box, cutoff, potential, atom_types)
    assert np.allclose(f_cell, f_ref, atol=1e-10, rtol=1e-10)

    pe_obs = compute_observables(r, v, 1.0, box, potential, cutoff, atom_types=atom_types)["PE"]
    pe_ref = _bruteforce_pe(r, box, cutoff, potential, atom_types)
    assert np.isclose(float(pe_obs), float(pe_ref), atol=1e-10, rtol=1e-10)


def test_morse_pair_coeff_matrix_matches_bruteforce():
    r = np.array(
        [
            [1.0, 0.9, 1.1],
            [2.4, 2.3, 1.9],
            [3.3, 2.9, 2.7],
        ],
        dtype=float,
    )
    v = np.zeros_like(r)
    box = 20.0
    cutoff = 8.0
    atom_types = np.array([1, 2, 1], dtype=np.int32)
    potential = make_potential(
        "morse",
        {
            "pair_coeffs": {
                "1-1": {"D_e": 0.29614, "a": 1.11892, "r0": 3.29692},
                "1-2": {"D_e": 0.25000, "a": 1.00000, "r0": 3.10000},
                "2-2": {"D_e": 0.28000, "a": 1.05000, "r0": 3.00000},
            }
        },
    )

    ids = np.arange(r.shape[0], dtype=np.int32)
    f_cell = forces_on_targets_celllist(
        r, box, potential, cutoff, ids, ids, rc=cutoff, atom_types=atom_types
    )
    f_ref = _bruteforce_forces(r, box, cutoff, potential, atom_types)
    assert np.allclose(f_cell, f_ref, atol=1e-10, rtol=1e-10)

    pe_obs = compute_observables(r, v, 1.0, box, potential, cutoff, atom_types=atom_types)["PE"]
    pe_ref = _bruteforce_pe(r, box, cutoff, potential, atom_types)
    assert np.isclose(float(pe_obs), float(pe_ref), atol=1e-10, rtol=1e-10)
