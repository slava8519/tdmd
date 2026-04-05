from __future__ import annotations

from types import SimpleNamespace

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
        rc: float | None = None,
        atom_types: np.ndarray,
        target_ids: np.ndarray,
        candidate_ids: np.ndarray,
    ) -> np.ndarray:
        del rc
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
        assert float(kwargs["rc"]) == 3.25
        return np.full((int(tids.size), 3), 3.0, dtype=float)

    monkeypatch.setattr(td_full_mpi, "try_gpu_forces_on_targets", _fake_gpu)
    out = wrapped.forces_on_targets(
        r=np.zeros((5, 3), dtype=float),
        box=10.0,
        cutoff=2.5,
        rc=3.25,
        atom_types=np.ones((5,), dtype=np.int32),
        target_ids=np.array([0, 1, 2], dtype=np.int32),
        candidate_ids=np.array([0, 1, 2, 3], dtype=np.int32),
    )
    assert np.allclose(out, 3.0)
    assert pot.cpu_calls == 0


def test_gpu_refinement_enables_runtime_dirty_tracking(monkeypatch):
    pot = _DummyPotential()
    backend = type("B", (), {"device": "cuda"})()
    wrapped = td_full_mpi._wrap_potential_for_gpu_refinement(potential=pot, backend=backend)
    seen: dict[str, bool] = {}

    def _fake_gpu(**kwargs):
        seen["prefer_marked_dirty"] = bool(kwargs.get("prefer_marked_dirty"))
        tids = np.asarray(kwargs["target_ids"], dtype=np.int32)
        return np.zeros((int(tids.size), 3), dtype=float)

    monkeypatch.setattr(td_full_mpi, "try_gpu_forces_on_targets", _fake_gpu)
    wrapped.forces_on_targets(
        r=np.zeros((4, 3), dtype=float),
        box=8.0,
        cutoff=2.5,
        atom_types=np.ones((4,), dtype=np.int32),
        target_ids=np.array([0, 1], dtype=np.int32),
        candidate_ids=np.array([0, 1, 2], dtype=np.int32),
    )
    assert seen == {"prefer_marked_dirty": True}


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


def test_finish_compute_marks_precompute_ids_dirty(monkeypatch):
    marked: list[np.ndarray | None] = []

    def _fake_mark(r: np.ndarray, ids=None):
        del r
        marked.append(None if ids is None else np.asarray(ids, dtype=np.int32).copy())

    monkeypatch.setattr(td_full_mpi, "mark_device_state_dirty", _fake_mark)
    zone = SimpleNamespace(
        atom_ids=np.array([1, 3], dtype=np.int32),
        ztype=td_full_mpi.ZoneType.W,
        step_id=0,
        halo_ids=np.empty((0,), dtype=np.int32),
    )

    class _FakeAutom:
        def __init__(self):
            self.work_zid = 0

        def compute_step_for_work_zone(self, **kwargs):
            del kwargs
            zone.atom_ids = np.array([1], dtype=np.int32)
            zone.ztype = td_full_mpi.ZoneType.S
            return 0

    td_full_mpi._finish_compute_with_trace_impl(
        None,
        _FakeAutom(),
        [zone],
        np.ones((5,), dtype=np.int32),
        r=np.zeros((5, 3), dtype=float),
        v=np.zeros((5, 3), dtype=float),
        mass=1.0,
        dt=0.005,
        potential=object(),
        cutoff=2.5,
        rc=2.5,
        skin_global=0.0,
        step=1,
        verlet_k_steps=1,
        backend=type("B", (), {"device": "cuda"})(),
    )

    assert len(marked) == 1
    assert np.array_equal(marked[0], np.array([1, 3], dtype=np.int32))


def test_handle_record_marks_received_ids_dirty_on_cuda(monkeypatch):
    marked: list[np.ndarray | None] = []

    def _fake_mark(r: np.ndarray, ids=None):
        del r
        marked.append(None if ids is None else np.asarray(ids, dtype=np.int32).copy())

    monkeypatch.setattr(td_full_mpi, "mark_device_state_dirty", _fake_mark)
    ctx = SimpleNamespace(
        zones=[SimpleNamespace(ztype=td_full_mpi.ZoneType.D)],
        r=np.zeros((4, 3), dtype=float),
        v=np.zeros((4, 3), dtype=float),
        backend=type("B", (), {"device": "cuda"})(),
        holder_map=[-1],
        holder_ver=[0],
    )
    ids = np.array([1, 2], dtype=np.int32)
    rr = np.full((2, 3), 1.25, dtype=float)
    vv = np.full((2, 3), -0.5, dtype=float)

    td_full_mpi._handle_record(
        ctx,
        src_rank=1,
        rectype=td_full_mpi.REC_HOLDER,
        subtype=0,
        zid=0,
        ids=ids,
        rr=rr,
        vv=vv,
        step_id=3,
    )

    assert np.allclose(ctx.r[ids], rr)
    assert np.allclose(ctx.v[ids], vv)
    assert len(marked) == 1
    assert np.array_equal(marked[0], ids)


def test_apply_ensemble_marks_device_state_when_box_rescales(monkeypatch):
    marked: list[np.ndarray | None] = []

    def _fake_mark(r: np.ndarray, ids=None):
        del r
        marked.append(None if ids is None else np.asarray(ids, dtype=np.int32).copy())

    def _fake_apply_ensemble_step(**kwargs):
        del kwargs
        return 10.1, 1.0, 1.01

    monkeypatch.setattr(td_full_mpi, "mark_device_state_dirty", _fake_mark)
    monkeypatch.setattr(td_full_mpi, "apply_ensemble_step", _fake_apply_ensemble_step)

    zones = [SimpleNamespace(z0=0.0, z1=10.0, n_cells=1)]
    autom = SimpleNamespace(box=10.0, zwidth=10.0)
    sim = td_full_mpi._TDMPISimState(box=10.0)

    td_full_mpi._apply_ensemble_impl(
        SimpleNamespace(kind="npt"),
        zones,
        autom,
        sim,
        r=np.zeros((3, 3), dtype=float),
        v=np.zeros((3, 3), dtype=float),
        mass=1.0,
        potential=object(),
        cutoff=2.5,
        atom_types=np.ones((3,), dtype=np.int32),
        dt=0.005,
        step=1,
        backend=type("B", (), {"device": "cuda"})(),
    )

    assert len(marked) == 1
    assert marked[0] is None
    assert sim.box == 10.1
