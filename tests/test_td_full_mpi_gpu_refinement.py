from __future__ import annotations

import numpy as np

import tdmd.td_full_mpi as td_full_mpi


class _DummyPotential:
    def __init__(self):
        self.cpu_calls = 0

    def forces_on_targets(
        self,
        *,
        r: np.ndarray,
        box: float,
        cutoff: float,
        atom_types: np.ndarray,
        target_ids: np.ndarray,
        candidate_ids: np.ndarray,
    ) -> np.ndarray:
        self.cpu_calls += 1
        return np.full((int(target_ids.size), 3), 7.0, dtype=float)


def test_wrap_potential_for_gpu_refinement_keeps_cpu_backend():
    pot = _DummyPotential()
    backend = type("B", (), {"device": "cpu"})()
    wrapped = td_full_mpi._wrap_potential_for_gpu_refinement(potential=pot, backend=backend)
    assert wrapped is pot


def test_gpu_refinement_prefers_gpu_force_path(monkeypatch):
    pot = _DummyPotential()
    backend = type("B", (), {"device": "cuda"})()
    wrapped = td_full_mpi._wrap_potential_for_gpu_refinement(potential=pot, backend=backend)

    def _fake_gpu(**kwargs):
        tids = np.asarray(kwargs["target_ids"], dtype=np.int32)
        return np.full((int(tids.size), 3), 3.0, dtype=float)

    monkeypatch.setattr(td_full_mpi, "try_gpu_forces_on_targets", _fake_gpu)
    out = wrapped.forces_on_targets(
        r=np.zeros((5, 3), dtype=float),
        box=10.0,
        cutoff=2.5,
        atom_types=np.ones((5,), dtype=np.int32),
        target_ids=np.array([0, 1, 2], dtype=np.int32),
        candidate_ids=np.array([0, 1, 2, 3], dtype=np.int32),
    )
    assert np.allclose(out, 3.0)
    assert pot.cpu_calls == 0


def test_gpu_refinement_falls_back_to_cpu_when_gpu_unavailable(monkeypatch):
    pot = _DummyPotential()
    backend = type("B", (), {"device": "cuda"})()
    wrapped = td_full_mpi._wrap_potential_for_gpu_refinement(potential=pot, backend=backend)

    monkeypatch.setattr(td_full_mpi, "try_gpu_forces_on_targets", lambda **kwargs: None)
    out = wrapped.forces_on_targets(
        r=np.zeros((5, 3), dtype=float),
        box=10.0,
        cutoff=2.5,
        atom_types=np.ones((5,), dtype=np.int32),
        target_ids=np.array([0, 1], dtype=np.int32),
        candidate_ids=np.array([0, 1, 2], dtype=np.int32),
    )
    assert np.allclose(out, 7.0)
    assert pot.cpu_calls == 1
