from __future__ import annotations

import numpy as np
import pytest

from tdmd.potentials import (
    ML_REFERENCE_CONTRACT_VERSION,
    MLReferencePotential,
    describe_ml_reference_contract,
    make_potential,
)


def _ml_reference_params() -> dict:
    return {
        "contract": {
            "version": ML_REFERENCE_CONTRACT_VERSION,
            "cutoff": {"radius": 2.2, "smoothing": "cosine"},
            "descriptor": {"family": "radial_density", "width": 0.75},
            "neighbor": {
                "mode": "candidate_local",
                "requires_full_system_barrier": False,
            },
            "inference": {
                "family": "quadratic_density",
                "cpu_reference": True,
                "target_local_supported": True,
            },
        },
        "species": [
            {"bias": 0.6, "quadratic": 0.4, "neighbor_weight": 1.0},
            {"bias": 0.8, "quadratic": 0.3, "neighbor_weight": 1.2},
        ],
    }


def _state() -> tuple[np.ndarray, np.ndarray]:
    r = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.9, 1.0, 1.0],
            [1.0, 1.8, 1.0],
        ],
        dtype=float,
    )
    atom_types = np.array([1, 2, 1], dtype=np.int32)
    return r, atom_types


def test_ml_reference_contract_is_versioned_and_exposed():
    pot = make_potential("ml/reference", _ml_reference_params())
    assert isinstance(pot, MLReferencePotential)
    contract = describe_ml_reference_contract(pot)
    assert contract is not None
    assert contract["version"] == ML_REFERENCE_CONTRACT_VERSION
    assert contract["cutoff"]["radius"] == pytest.approx(2.2)
    assert contract["cutoff"]["smoothing"] == "cosine"
    assert contract["descriptor"]["family"] == "radial_density"
    assert contract["neighbor"]["mode"] == "candidate_local"
    assert contract["neighbor"]["requires_full_system_barrier"] is False
    assert contract["inference"]["family"] == "quadratic_density"
    assert contract["inference"]["cpu_reference"] is True
    assert contract["inference"]["target_local_supported"] is True


def test_ml_reference_target_local_matches_full_system_slice():
    pot = make_potential("ml/reference", _ml_reference_params())
    r, atom_types = _state()
    forces_full, pe, virial = pot.forces_energy_virial(
        r, box=10.0, cutoff=3.0, atom_types=atom_types
    )
    assert np.isfinite(pe)
    assert np.isfinite(virial)
    forces_target = pot.forces_on_targets(
        r,
        box=10.0,
        cutoff=3.0,
        atom_types=atom_types,
        target_ids=np.array([0, 2], dtype=np.int32),
        candidate_ids=np.array([1], dtype=np.int32),
    )
    assert np.allclose(forces_target, forces_full[[0, 2]], atol=1e-12, rtol=1e-12)


def test_ml_reference_rejects_runtime_cutoff_smaller_than_contract():
    pot = make_potential("ml/reference", _ml_reference_params())
    r, atom_types = _state()
    with pytest.raises(
        ValueError, match="runtime cutoff is smaller than ml/reference contract cutoff"
    ):
        pot.forces_energy_virial(r, box=10.0, cutoff=1.5, atom_types=atom_types)
