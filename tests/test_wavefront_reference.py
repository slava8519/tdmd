from __future__ import annotations

import numpy as np

from tdmd.potentials import make_potential
from tdmd.wavefront_reference import (
    WAVEFRONT_REFERENCE_CONTRACT_VERSION,
    WAVEFRONT_REFERENCE_KIND_MANY_BODY_TARGET_LOCAL,
    WAVEFRONT_REFERENCE_KIND_PAIR_SYNC_1D,
    WAVEFRONT_REFERENCE_KIND_SHADOW_WAVE_BATCH,
    prove_wavefront_1d_reference_equivalence,
)


def _dense_alloy_state(
    *,
    n_atoms: int,
    lattice_a: float,
    seed: int,
    jitter: float,
    velocity_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    basis = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        dtype=float,
    )
    n_cells = max(1, int(np.ceil((float(n_atoms) / 4.0) ** (1.0 / 3.0))))
    box = float(n_cells) * float(lattice_a)
    pts: list[np.ndarray] = []
    for ix in range(n_cells):
        ox = float(ix) * float(lattice_a)
        for iy in range(n_cells):
            oy = float(iy) * float(lattice_a)
            for iz in range(n_cells):
                oz = float(iz) * float(lattice_a)
                pts.append((basis * float(lattice_a)) + np.asarray([ox, oy, oz], dtype=float))
    r0 = np.vstack(pts)[: int(n_atoms)].astype(np.float64, copy=True)
    rng = np.random.default_rng(int(seed))
    if float(jitter) > 0.0:
        r0 += rng.normal(0.0, float(jitter), size=r0.shape)
        r0[:] = np.mod(r0, box)
    atom_types = np.ones((int(n_atoms),), dtype=np.int32)
    atom_types[rng.permutation(int(n_atoms))[: int(n_atoms) // 2]] = 2
    masses = np.where(atom_types == 1, 26.9815385, 63.5460).astype(np.float64)
    v0 = rng.normal(0.0, float(velocity_std), size=(int(n_atoms), 3)).astype(np.float64)
    v0 -= np.mean(v0, axis=0, keepdims=True)
    return r0, v0, masses, atom_types, box


def test_wavefront_reference_pair_shadow_matches_sync_reference():
    potential = make_potential(
        "morse",
        {
            "D_e": 0.29614,
            "a": 1.11892,
            "r0": 3.29692,
        },
    )
    r0 = np.asarray(
        [
            [1.0, 1.0, 5.0],
            [1.5, 1.0, 5.0],
            [1.0, 1.0, 15.0],
            [1.5, 1.0, 15.0],
            [1.0, 1.0, 25.0],
            [1.5, 1.0, 25.0],
            [1.0, 1.0, 35.0],
            [1.5, 1.0, 35.0],
        ],
        dtype=float,
    )
    out = prove_wavefront_1d_reference_equivalence(
        r0=r0,
        v0=np.zeros_like(r0),
        mass=44.80137,
        box=40.0,
        potential=potential,
        dt=0.001,
        cutoff=4.0,
        n_steps=2,
        atom_types=np.ones((r0.shape[0],), dtype=np.int32),
        cell_size=10.0,
        zones_total=4,
        zone_cells_w=1,
        zone_cells_s=1,
    )

    assert bool(out.get("ok_all")) is True
    assert str(out.get("contract_version", "")) == WAVEFRONT_REFERENCE_CONTRACT_VERSION
    assert str(out.get("reference_force_kind", "")) == WAVEFRONT_REFERENCE_KIND_PAIR_SYNC_1D
    assert str(out.get("shadow_force_kind", "")) == WAVEFRONT_REFERENCE_KIND_SHADOW_WAVE_BATCH
    assert bool(out.get("multi_zone_wave_seen")) is True
    assert int(out.get("max_wave_size", 0)) == 2
    assert float(out.get("max_force_max_abs", 1.0)) == 0.0


def test_wavefront_reference_many_body_shadow_matches_sequential_target_local():
    potential = make_potential(
        "eam/alloy",
        {
            "file": "examples/potentials/eam_alloy/AlCu.eam.alloy",
            "elements": ["Al", "Cu"],
        },
    )
    r0 = np.asarray(
        [
            [1.0, 1.0, 8.0],
            [1.8, 1.0, 8.6],
            [1.0, 1.0, 12.0],
            [1.8, 1.0, 12.6],
            [1.0, 1.0, 18.0],
            [1.8, 1.0, 18.6],
            [1.0, 1.0, 22.0],
            [1.8, 1.0, 22.6],
        ],
        dtype=float,
    )
    atom_types = np.asarray([1, 2, 1, 2, 1, 2, 1, 2], dtype=np.int32)
    masses = np.where(atom_types == 1, 26.9815385, 63.5460)
    out = prove_wavefront_1d_reference_equivalence(
        r0=r0,
        v0=np.zeros_like(r0),
        mass=masses,
        box=40.0,
        potential=potential,
        dt=0.001,
        cutoff=6.5,
        n_steps=2,
        atom_types=atom_types,
        cell_size=10.0,
        zones_total=4,
        zone_cells_w=1,
        zone_cells_s=1,
    )

    assert bool(out.get("ok_all")) is True
    assert str(out.get("contract_version", "")) == WAVEFRONT_REFERENCE_CONTRACT_VERSION
    assert (
        str(out.get("reference_force_kind", "")) == WAVEFRONT_REFERENCE_KIND_MANY_BODY_TARGET_LOCAL
    )
    assert str(out.get("shadow_force_kind", "")) == WAVEFRONT_REFERENCE_KIND_SHADOW_WAVE_BATCH
    assert bool(out.get("many_body")) is True
    assert bool(out.get("multi_zone_wave_seen")) is True
    assert int(out.get("max_wave_size", 0)) == 2
    assert float(out.get("max_force_max_abs", 1.0)) == 0.0


def test_wavefront_reference_many_body_dense_shadow_matches_sequential_target_local():
    potential = make_potential(
        "eam/alloy",
        {
            "file": "examples/potentials/eam_alloy/AlCu.eam.alloy",
            "elements": ["Al", "Cu"],
        },
    )
    r0, v0, masses, atom_types, box = _dense_alloy_state(
        n_atoms=256,
        lattice_a=4.05,
        seed=42,
        jitter=0.02,
        velocity_std=0.01,
    )
    out = prove_wavefront_1d_reference_equivalence(
        r0=r0,
        v0=v0,
        mass=masses,
        box=box,
        potential=potential,
        dt=0.001,
        cutoff=6.5,
        n_steps=1,
        atom_types=atom_types,
        cell_size=2.7,
        zones_total=2,
        zone_cells_w=1,
        zone_cells_s=1,
        require_multi_zone_wave=False,
    )

    assert bool(out.get("ok_all")) is True
    assert (
        str(out.get("reference_force_kind", "")) == WAVEFRONT_REFERENCE_KIND_MANY_BODY_TARGET_LOCAL
    )
    assert str(out.get("shadow_force_kind", "")) == WAVEFRONT_REFERENCE_KIND_SHADOW_WAVE_BATCH
    assert bool(out.get("many_body")) is True
    assert float(out.get("max_force_max_abs", 1.0)) == 0.0
