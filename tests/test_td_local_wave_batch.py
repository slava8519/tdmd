from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import tdmd.td_local as td_local
from tdmd.celllist import forces_on_targets_celllist
from tdmd.potentials import make_potential
from tdmd.td_local import run_td_local


class _TraceRecorder:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def log(self, **kwargs) -> None:
        self.events.append(dict(kwargs))


class _ZeroManyBodyPotential:
    def __init__(self) -> None:
        self.full_calls = 0

    def forces_energy_virial(self, r, box, cutoff, atom_types):
        del box, cutoff, atom_types
        self.full_calls += 1
        return np.zeros_like(r, dtype=float), 0.0, 0.0

    def forces_on_targets(self, r, box, cutoff, atom_types, target_ids, candidate_ids, rc=None):
        del r, box, cutoff, atom_types, candidate_ids, rc
        return np.zeros((int(np.asarray(target_ids, dtype=np.int32).size), 3), dtype=float)


def _wave_state() -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    r = np.asarray(
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
    v = np.zeros_like(r)
    atom_types = np.ones((r.shape[0],), dtype=np.int32)
    return r, v, 40.0, atom_types


def _assert_trace_keeps_one_zone_in_flight(events: list[dict[str, object]]) -> None:
    in_flight = 0
    for event in events:
        kind = str(event.get("event", ""))
        if kind == "START_COMPUTE":
            assert in_flight == 0
            in_flight = 1
        elif kind == "FINISH_COMPUTE":
            assert in_flight == 1
            in_flight = 0
    assert in_flight == 0


def test_td_local_async_1d_wave_batch_reduces_pair_gpu_launches_without_trace_overlap(monkeypatch):
    backend = SimpleNamespace(device="cuda", xp=np, cuda_available=True, reason="test")
    monkeypatch.setattr(td_local, "resolve_backend", lambda device="auto": backend)
    monkeypatch.setattr(td_local, "mark_device_state_dirty", lambda r, ids=None: None)

    pair_calls_baseline: list[np.ndarray] = []
    pair_calls_batched: list[np.ndarray] = []
    potential = make_potential(
        "morse",
        {
            "D_e": 0.29614,
            "a": 1.11892,
            "r0": 3.29692,
        },
    )

    def _fake_pair_backend_factory(calls: list[np.ndarray]):
        def _fake_pair_backend(
            *,
            r,
            box,
            cutoff,
            potential,
            target_ids,
            candidate_ids,
            atom_types,
            backend,
            prefer_marked_dirty=False,
        ):
            del backend, prefer_marked_dirty
            tids = np.asarray(target_ids, dtype=np.int32)
            cids = np.asarray(candidate_ids, dtype=np.int32)
            calls.append(tids.copy())
            return forces_on_targets_celllist(
                np.asarray(r, dtype=np.float64),
                float(box),
                potential,
                float(cutoff),
                tids,
                cids,
                rc=float(cutoff),
                atom_types=np.asarray(atom_types, dtype=np.int32),
            )

        return _fake_pair_backend

    orig_build_wave_batch_state = td_local._build_async_1d_wave_batch_state

    r_base, v_base, box, atom_types = _wave_state()
    monkeypatch.setattr(
        td_local,
        "forces_on_targets_pair_backend",
        _fake_pair_backend_factory(pair_calls_baseline),
    )
    monkeypatch.setattr(td_local, "_build_async_1d_wave_batch_state", lambda **kwargs: None)
    run_td_local(
        r_base,
        v_base,
        44.80137,
        box,
        potential,
        0.001,
        4.0,
        1,
        atom_types=atom_types,
        cell_size=10.0,
        zones_total=4,
        zone_cells_w=1,
        zone_cells_s=1,
        decomposition="1d",
        sync_mode=False,
        strict_min_zone_width=True,
        ensemble_kind="nve",
        device="cuda",
    )
    diag_baseline = td_local.get_last_td_local_wave_batch_diagnostics()

    monkeypatch.setattr(td_local, "_build_async_1d_wave_batch_state", orig_build_wave_batch_state)
    r_wave, v_wave, box, atom_types = _wave_state()
    trace = _TraceRecorder()
    monkeypatch.setattr(
        td_local,
        "forces_on_targets_pair_backend",
        _fake_pair_backend_factory(pair_calls_batched),
    )
    run_td_local(
        r_wave,
        v_wave,
        44.80137,
        box,
        potential,
        0.001,
        4.0,
        1,
        atom_types=atom_types,
        cell_size=10.0,
        zones_total=4,
        zone_cells_w=1,
        zone_cells_s=1,
        decomposition="1d",
        sync_mode=False,
        strict_min_zone_width=True,
        ensemble_kind="nve",
        device="cuda",
        trace=trace,
    )
    diag_wave = td_local.get_last_td_local_wave_batch_diagnostics()

    assert np.allclose(r_wave, r_base, atol=1e-12, rtol=1e-12)
    assert np.allclose(v_wave, v_base, atol=1e-12, rtol=1e-12)
    assert len(pair_calls_batched) < len(pair_calls_baseline)
    assert max(int(call.size) for call in pair_calls_batched) > 2
    assert diag_baseline.version == td_local.TD_LOCAL_WAVE_BATCH_DIAGNOSTICS_VERSION
    assert bool(diag_baseline.eligible) is True
    assert bool(diag_baseline.enabled) is False
    assert int(diag_baseline.successful_wave_batches) == 0
    assert int(diag_wave.successful_wave_batches) >= 1
    assert bool(diag_wave.enabled) is True
    assert float(diag_wave.launches_saved_per_step) > 0.0
    assert int(diag_wave.estimated_pre_force_launches_saved_total) >= 1
    assert int(diag_wave.cached_pre_force_hits) >= 1
    _assert_trace_keeps_one_zone_in_flight(trace.events)


def test_td_local_async_1d_wave_batch_reduces_many_body_gpu_launches(monkeypatch):
    backend = SimpleNamespace(device="cuda", xp=np, cuda_available=True, reason="test")
    monkeypatch.setattr(td_local, "resolve_backend", lambda device="auto": backend)
    monkeypatch.setattr(td_local, "mark_device_state_dirty", lambda r, ids=None: None)

    gpu_calls_baseline: list[np.ndarray] = []
    gpu_calls_batched: list[np.ndarray] = []
    potential = _ZeroManyBodyPotential()

    def _fake_gpu_factory(calls: list[np.ndarray]):
        def _fake_gpu(**kwargs):
            tids = np.asarray(kwargs["target_ids"], dtype=np.int32)
            calls.append(tids.copy())
            return np.zeros((int(tids.size), 3), dtype=float)

        return _fake_gpu

    orig_build_wave_batch_state = td_local._build_async_1d_wave_batch_state

    r_base, v_base, box, atom_types = _wave_state()
    monkeypatch.setattr(
        td_local, "try_gpu_forces_on_targets", _fake_gpu_factory(gpu_calls_baseline)
    )
    monkeypatch.setattr(td_local, "_build_async_1d_wave_batch_state", lambda **kwargs: None)
    run_td_local(
        r_base,
        v_base,
        1.0,
        box,
        potential,
        0.001,
        4.0,
        1,
        atom_types=atom_types,
        cell_size=10.0,
        zones_total=4,
        zone_cells_w=1,
        zone_cells_s=1,
        decomposition="1d",
        sync_mode=False,
        strict_min_zone_width=True,
        ensemble_kind="nve",
        device="cuda",
    )

    monkeypatch.setattr(td_local, "_build_async_1d_wave_batch_state", orig_build_wave_batch_state)
    r_wave, v_wave, box, atom_types = _wave_state()
    monkeypatch.setattr(td_local, "try_gpu_forces_on_targets", _fake_gpu_factory(gpu_calls_batched))
    run_td_local(
        r_wave,
        v_wave,
        1.0,
        box,
        potential,
        0.001,
        4.0,
        1,
        atom_types=atom_types,
        cell_size=10.0,
        zones_total=4,
        zone_cells_w=1,
        zone_cells_s=1,
        decomposition="1d",
        sync_mode=False,
        strict_min_zone_width=True,
        ensemble_kind="nve",
        device="cuda",
    )

    assert np.allclose(r_wave, r_base, atol=1e-12, rtol=1e-12)
    assert np.allclose(v_wave, v_base, atol=1e-12, rtol=1e-12)
    assert potential.full_calls == 0
    assert len(gpu_calls_batched) < len(gpu_calls_baseline)
    assert max(int(call.size) for call in gpu_calls_batched) > 2


def test_describe_td_local_wave_batch_contract_reports_cuda_async_1d_only():
    contract = td_local.describe_td_local_wave_batch_contract(
        sync_mode=False,
        decomposition="1d",
        device="cuda",
    )
    assert contract is not None
    assert contract["version"] == td_local.TD_LOCAL_WAVE_BATCH_CONTRACT_VERSION
    assert contract["wavefront_contract_version"] == "pr_sw01_v1"
    assert contract["diagnostics_contract_version"] == td_local.TD_LOCAL_WAVE_BATCH_DIAGNOSTICS_VERSION
    assert bool(contract["pre_force_batching"]) is True
    assert bool(contract["post_force_batching"]) is False
    assert bool(contract["formal_core_w_leq_1"]) is True

    assert td_local.describe_td_local_wave_batch_contract(device="cpu") is None
    assert td_local.describe_td_local_wave_batch_contract(decomposition="3d", device="cuda") is None
    assert td_local.describe_td_local_wave_batch_contract(sync_mode=True, device="cuda") is None


def test_reset_td_local_wave_batch_diagnostics_restores_default_snapshot():
    td_local.reset_td_local_wave_batch_diagnostics()
    diag = td_local.get_last_td_local_wave_batch_diagnostics()

    assert diag.version == td_local.TD_LOCAL_WAVE_BATCH_DIAGNOSTICS_VERSION
    assert bool(diag.eligible) is False
    assert bool(diag.enabled) is False
    assert int(diag.successful_wave_batches) == 0
