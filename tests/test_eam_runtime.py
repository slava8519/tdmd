from __future__ import annotations

import json
import numpy as np

from tdmd.potentials import EAMAlloyPotential, make_potential
from tdmd.serial import run_serial


def _two_atom_state(dist: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0 + float(dist), 1.0, 1.0],
        ],
        dtype=float,
    )


def test_eam_single_element_reference_values():
    pot = make_potential(
        "eam/alloy",
        {"file": "examples/interop/eam_al_ni_demo.setfl", "elements": ["Al"]},
    )
    assert isinstance(pot, EAMAlloyPotential)
    r = _two_atom_state(1.0)
    types = np.array([1, 1], dtype=np.int32)
    f, pe, vir = pot.forces_energy_virial(r, box=10.0, cutoff=5.0, atom_types=types)

    # Synthetic setfl expected values at r=1:
    # rho_i=1, rho_j=1, F=rho -> 2.0
    # phi_AlAl(1)=1.0 -> pair 1.0
    # total PE = 3.0
    # dE/dr = dphi + dF_i*drho_j + dF_j*drho_i = (-1)+(-1)+(-1) = -3
    # force magnitude = -dE/dr = 3
    assert np.isclose(pe, 3.0, atol=1e-12)
    assert np.isclose(vir, 3.0, atol=1e-12)
    assert np.allclose(f[0], np.array([-3.0, 0.0, 0.0]), atol=1e-12)
    assert np.allclose(f[1], np.array([3.0, 0.0, 0.0]), atol=1e-12)


def test_eam_alloy_reference_values():
    pot = make_potential(
        "eam_alloy",
        {"file": "examples/interop/eam_al_ni_demo.setfl", "elements": ["Al", "Ni"]},
    )
    assert isinstance(pot, EAMAlloyPotential)
    r = _two_atom_state(1.0)
    types = np.array([1, 2], dtype=np.int32)
    f, pe, vir = pot.forces_energy_virial(r, box=10.0, cutoff=5.0, atom_types=types)

    # Synthetic setfl expected values at r=1:
    # rho_Al<-Ni = 0.5, rho_Ni<-Al = 1.0
    # F_Al=1*rho=0.5, F_Ni=2*rho=2.0, phi_AlNi(1)=0.5
    # total PE = 3.0
    # dE/dr = dphi(-0.5) + dF_Al*drho_Ni(-0.5) + dF_Ni*drho_Al(-1.0) = -3.0
    # force magnitude = 3.0
    assert np.isclose(pe, 3.0, atol=1e-12)
    assert np.isclose(vir, 3.0, atol=1e-12)
    assert np.allclose(f[0], np.array([-3.0, 0.0, 0.0]), atol=1e-12)
    assert np.allclose(f[1], np.array([3.0, 0.0, 0.0]), atol=1e-12)


def test_run_serial_uses_eam_backend_and_preserves_total_momentum():
    pot = make_potential(
        "eam/alloy",
        {"file": "examples/interop/eam_al_ni_demo.setfl", "elements": ["Al", "Ni"]},
    )
    r = _two_atom_state(1.0)
    v = np.zeros_like(r)
    mass = np.array([26.9815386, 58.6934], dtype=float)
    types = np.array([1, 2], dtype=np.int32)
    run_serial(r, v, mass, box=10.0, potential=pot, dt=1e-3, cutoff=5.0, n_steps=4, atom_types=types)

    p_total = (mass[:, None] * v).sum(axis=0)
    assert np.all(np.isfinite(r))
    assert np.all(np.isfinite(v))
    assert np.allclose(p_total, np.zeros(3), atol=1e-10)


def test_eam_matches_lammps_reference_fixture():
    with open("examples/interop/eam_lammps_ref_demo.json", "r", encoding="utf-8") as f:
        ref = json.load(f)
    thr = ref["thresholds"]
    for case in ref["cases"]:
        pot = make_potential(
            "eam/alloy",
            {"file": "examples/interop/eam_al_ni_demo.setfl", "elements": case["elements"]},
        )
        r = np.asarray(case["positions"], dtype=float)
        types = np.asarray(case["atom_types"], dtype=np.int32)
        f_calc, pe_calc, vir_calc = pot.forces_energy_virial(
            r, box=float(case["box"]), cutoff=float(case["cutoff"]), atom_types=types
        )
        f_ref = np.asarray(case["expected_forces"], dtype=float)
        assert np.allclose(f_calc, f_ref, atol=float(thr["force_abs"]), rtol=0.0), case["name"]
        assert abs(float(pe_calc) - float(case["expected_pe"])) <= float(thr["energy_abs"]), case["name"]
        assert abs(float(vir_calc) - float(case["expected_virial"])) <= float(thr["virial_abs"]), case["name"]
