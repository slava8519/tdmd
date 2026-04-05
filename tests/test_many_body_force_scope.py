from __future__ import annotations

import numpy as np

import tdmd.td_local as td_local_mod
from tdmd.many_body_scope import td_local_many_body_force_scope
from tdmd.td_local import describe_many_body_force_scope, run_td_local


class _StubManyBodyPotential:
    def __init__(self) -> None:
        self.full_calls = 0
        self.target_calls = 0

    def forces_energy_virial(self, r, box, cutoff, atom_types):
        self.full_calls += 1
        return np.zeros_like(r, dtype=float), 0.0, 0.0

    def forces_on_targets(self, r, box, cutoff, atom_types, target_ids, candidate_ids, rc=None):
        del box, cutoff, atom_types, candidate_ids, rc
        self.target_calls += 1
        return np.zeros((np.asarray(target_ids, dtype=np.int32).size, 3), dtype=float)


def test_td_local_many_body_force_scope_async_1d_cuda_reports_target_local_target_slice():
    scope = td_local_many_body_force_scope(
        _StubManyBodyPotential(),
        sync_mode=False,
        decomposition="1d",
        device="cuda",
    )
    assert scope is not None
    assert scope.runtime_kind == "td_local.async_1d"
    assert scope.evaluation_scope == "target_local"
    assert scope.consumption_scope == "target_ids"
    assert scope.target_local_available is True


def test_describe_many_body_force_scope_sync_global_reports_full_system_all_atoms():
    scope = describe_many_body_force_scope(
        _StubManyBodyPotential(),
        sync_mode=True,
        decomposition="1d",
        device="cpu",
    )
    assert scope is not None
    assert scope["runtime_kind"] == "td_local.sync_global"
    assert scope["evaluation_scope"] == "full_system"
    assert scope["consumption_scope"] == "all_atoms"
    assert bool(scope["target_local_available"]) is True


def test_td_local_many_body_force_scope_async_cpu_reports_target_local():
    scope = describe_many_body_force_scope(
        _StubManyBodyPotential(),
        sync_mode=False,
        decomposition="1d",
        device="cpu",
    )
    assert scope is not None
    assert scope["runtime_kind"] == "td_local.async_1d"
    assert scope["evaluation_scope"] == "target_local"
    assert scope["consumption_scope"] == "target_ids"
    assert bool(scope["target_local_available"]) is True


def test_td_local_async_many_body_cpu_uses_target_local_not_forces_full():
    potential = _StubManyBodyPotential()
    r = np.array(
        [
            [0.2, 0.2, 0.2],
            [1.2, 0.2, 0.2],
            [2.2, 0.2, 0.2],
            [3.2, 0.2, 0.2],
        ],
        dtype=float,
    )
    v = np.zeros_like(r)

    run_td_local(
        r,
        v,
        1.0,
        8.0,
        potential,
        0.001,
        2.5,
        1,
        atom_types=np.ones((r.shape[0],), dtype=np.int32),
        cell_size=1.0,
        zones_total=2,
        zone_cells_w=1,
        zone_cells_s=2,
        decomposition="1d",
        sync_mode=False,
        strict_min_zone_width=True,
        ensemble_kind="nve",
        device="cpu",
    )

    assert potential.full_calls == 0
    assert potential.target_calls > 0


def test_td_local_async_3d_many_body_cpu_uses_target_local_not_forces_full():
    potential = _StubManyBodyPotential()
    r = np.array(
        [
            [0.2, 0.2, 0.2],
            [1.2, 0.2, 0.2],
            [4.2, 0.2, 0.2],
            [5.2, 0.2, 0.2],
        ],
        dtype=float,
    )
    v = np.zeros_like(r)

    run_td_local(
        r,
        v,
        1.0,
        8.0,
        potential,
        0.001,
        2.5,
        1,
        atom_types=np.ones((r.shape[0],), dtype=np.int32),
        cell_size=1.0,
        zones_total=2,
        zone_cells_w=1,
        zone_cells_s=2,
        decomposition="3d",
        zones_nx=2,
        zones_ny=1,
        zones_nz=1,
        sync_mode=False,
        strict_min_zone_width=True,
        ensemble_kind="nve",
        device="cpu",
    )

    assert potential.full_calls == 0
    assert potential.target_calls > 0


def test_forces_many_body_targets_cuda_uses_target_local_gpu_dispatch(monkeypatch):
    calls = {"gpu": 0, "full": 0}

    def _fake_gpu(**kwargs):
        calls["gpu"] += 1
        tids = np.asarray(kwargs["target_ids"], dtype=np.int32)
        return np.zeros((tids.size, 3), dtype=float)

    backend = type("Backend", (), {"device": "cuda"})()
    ctx = type("Ctx", (), {})()
    ctx.backend = backend
    ctx.r = np.zeros((4, 3), dtype=float)
    ctx.box = 8.0
    ctx.cutoff = 2.5
    ctx.potential = _StubManyBodyPotential()
    ctx.atom_types = np.ones((4,), dtype=np.int32)

    def _forces_full(_rr):
        calls["full"] += 1
        return np.zeros((4, 3), dtype=float)

    ctx.forces_full = _forces_full

    monkeypatch.setattr(td_local_mod, "try_gpu_forces_on_targets", _fake_gpu)
    out = td_local_mod._forces_many_body_targets(
        ctx,
        np.array([0, 1], dtype=np.int32),
        np.array([0, 1, 2], dtype=np.int32),
        2.5,
    )

    assert out.shape == (2, 3)
    assert calls["gpu"] == 1
    assert calls["full"] == 0
