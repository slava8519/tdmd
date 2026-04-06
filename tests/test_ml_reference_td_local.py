from __future__ import annotations

import numpy as np

import tdmd.td_local as td_local_mod
from tdmd.potentials import ML_REFERENCE_CONTRACT_VERSION, make_potential
from tdmd.serial import run_serial
from tdmd.td_local import run_td_local


def _ml_reference_params() -> dict:
    return {
        "contract": {
            "version": ML_REFERENCE_CONTRACT_VERSION,
            "cutoff": {"radius": 2.4, "smoothing": "cosine"},
            "descriptor": {"family": "radial_density", "width": 0.85},
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
            {"bias": 0.4, "quadratic": 0.5, "neighbor_weight": 1.0},
            {"bias": 0.6, "quadratic": 0.3, "neighbor_weight": 1.1},
        ],
    }


def _state():
    r = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [1.0, 2.1, 1.0],
            [2.0, 2.0, 1.0],
        ],
        dtype=float,
    )
    v = np.zeros_like(r)
    masses = np.array([1.0, 1.5, 1.0, 1.5], dtype=float)
    atom_types = np.array([1, 2, 1, 2], dtype=np.int32)
    return r, v, masses, atom_types


def _run_serial_vs_td_local(*, sync_mode: bool, zones_total: int):
    r0, v0, masses, atom_types = _state()
    pot = make_potential("ml/reference", _ml_reference_params())
    r_serial = r0.copy()
    v_serial = v0.copy()
    run_serial(
        r_serial,
        v_serial,
        masses,
        box=10.0,
        potential=pot,
        dt=1e-3,
        cutoff=3.0,
        n_steps=4,
        atom_types=atom_types,
    )
    r_td = r0.copy()
    v_td = v0.copy()
    run_td_local(
        r_td,
        v_td,
        masses,
        10.0,
        pot,
        1e-3,
        3.0,
        n_steps=4,
        atom_types=atom_types,
        cell_size=1.5,
        zones_total=int(zones_total),
        zone_cells_w=1,
        zone_cells_s=1,
        traversal="forward",
        use_verlet=False,
        decomposition="1d",
        sync_mode=bool(sync_mode),
    )
    return r_serial, v_serial, r_td, v_td


def _run_td_local_case(*, decomposition: str, zones_total: int, zones_nx: int = 1):
    r0, v0, masses, atom_types = _state()
    pot = make_potential("ml/reference", _ml_reference_params())
    r = r0.copy()
    v = v0.copy()
    run_td_local(
        r,
        v,
        masses,
        10.0,
        pot,
        1e-3,
        3.0,
        n_steps=4,
        atom_types=atom_types,
        cell_size=1.5,
        zones_total=int(zones_total),
        zone_cells_w=1,
        zone_cells_s=1,
        traversal="forward",
        use_verlet=False,
        decomposition=str(decomposition),
        sync_mode=False,
        zones_nx=int(zones_nx),
        zones_ny=1,
        zones_nz=1,
    )
    return r, v


def test_td_local_ml_reference_sync_mode_matches_serial():
    r_s, v_s, r_t, v_t = _run_serial_vs_td_local(sync_mode=True, zones_total=2)
    assert np.allclose(r_s, r_t, atol=1e-12, rtol=1e-12)
    assert np.allclose(v_s, v_t, atol=1e-12, rtol=1e-12)


def test_td_local_ml_reference_async_target_local_1d_matches_legacy_full_system(monkeypatch):
    r_target, v_target = _run_td_local_case(decomposition="1d", zones_total=2)
    monkeypatch.setattr(
        td_local_mod,
        "_forces_many_body_targets",
        lambda ctx, target_ids, candidate_ids, rc: ctx.forces_full(ctx.r)[target_ids],
    )
    r_legacy, v_legacy = _run_td_local_case(decomposition="1d", zones_total=2)
    assert np.allclose(r_target, r_legacy, atol=1e-12, rtol=1e-12)
    assert np.allclose(v_target, v_legacy, atol=1e-12, rtol=1e-12)


def test_td_local_ml_reference_async_target_local_3d_matches_legacy_full_system(monkeypatch):
    r_target, v_target = _run_td_local_case(decomposition="3d", zones_total=2, zones_nx=2)
    monkeypatch.setattr(
        td_local_mod,
        "_forces_many_body_targets",
        lambda ctx, target_ids, candidate_ids, rc: ctx.forces_full(ctx.r)[target_ids],
    )
    r_legacy, v_legacy = _run_td_local_case(decomposition="3d", zones_total=2, zones_nx=2)
    assert np.allclose(r_target, r_legacy, atol=1e-12, rtol=1e-12)
    assert np.allclose(v_target, v_legacy, atol=1e-12, rtol=1e-12)
