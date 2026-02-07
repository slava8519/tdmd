from __future__ import annotations

import warnings

import pytest

from tdmd.backend import cuda_device_for_local_rank, local_rank_from_env, resolve_backend
from tdmd.config import load_config


def test_resolve_backend_cpu_forced():
    b = resolve_backend("cpu")
    assert b.device == "cpu"


def test_resolve_backend_invalid_device():
    with pytest.raises(ValueError, match="device must be one of"):
        resolve_backend("tpu")


def test_resolve_backend_cuda_fallback_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        b = resolve_backend("cuda")
    assert b.device in ("cpu", "cuda")
    if b.device == "cpu":
        assert any("falling back to CPU" in str(x.message) for x in w)


def test_resolve_backend_auto_force_cpu_env(monkeypatch):
    monkeypatch.setenv("TDMD_FORCE_CPU", "1")
    b = resolve_backend("auto")
    assert b.device == "cpu"


def test_load_config_rejects_invalid_run_device(tmp_path):
    cfg_text = """
system:
  n_atoms: 8
  mass: 1.0
  box: 10.0
  temperature: 1.0
  seed: 1
potential:
  kind: lj
  params:
    epsilon: 1.0
    sigma: 1.0
run:
  dt: 0.005
  n_steps: 10
  thermo_every: 5
  cutoff: 2.5
  device: tpu
td:
  cell_size: 1.0
  zones_total: 2
  zone_cells_w: 1
  zone_cells_s: 1
  traversal: snake
"""
    path = tmp_path / "bad_device.yaml"
    path.write_text(cfg_text, encoding="utf-8")
    with pytest.raises(ValueError, match="run.device"):
        load_config(str(path))


def test_local_rank_from_env_priority():
    env = {
        "OMPI_COMM_WORLD_LOCAL_RANK": "3",
        "MPI_LOCALRANKID": "2",
        "SLURM_LOCALID": "1",
    }
    assert local_rank_from_env(env) == 3
    env2 = {"MPI_LOCALRANKID": "4"}
    assert local_rank_from_env(env2) == 4
    env3 = {"SLURM_LOCALID": "5"}
    assert local_rank_from_env(env3) == 5
    assert local_rank_from_env({}) == 0


def test_cuda_device_for_local_rank_round_robin():
    assert cuda_device_for_local_rank(0, 2) == 0
    assert cuda_device_for_local_rank(1, 2) == 1
    assert cuda_device_for_local_rank(2, 2) == 0
    assert cuda_device_for_local_rank(5, 2) == 1
    with pytest.raises(ValueError, match="device_count"):
        cuda_device_for_local_rank(0, 0)


def test_load_config_parses_mpi_overlap_flags(tmp_path):
    cfg_text = """
system:
  n_atoms: 8
  mass: 1.0
  box: 10.0
  temperature: 1.0
  seed: 1
potential:
  kind: lj
  params:
    epsilon: 1.0
    sigma: 1.0
run:
  dt: 0.005
  n_steps: 10
  thermo_every: 5
  cutoff: 2.5
td:
  cell_size: 1.0
  zones_total: 2
  zone_cells_w: 1
  zone_cells_s: 1
  traversal: snake
  cuda_aware_mpi: true
  comm_overlap_isend: true
"""
    path = tmp_path / "mpi_overlap.yaml"
    path.write_text(cfg_text, encoding="utf-8")
    cfg = load_config(str(path))
    assert cfg.td.cuda_aware_mpi is True
    assert cfg.td.comm_overlap_isend is True


def test_load_config_defaults_to_nve_ensemble(tmp_path):
    cfg_text = """
system:
  n_atoms: 8
  mass: 1.0
  box: 10.0
  temperature: 1.0
  seed: 1
potential:
  kind: lj
  params:
    epsilon: 1.0
    sigma: 1.0
run:
  dt: 0.005
  n_steps: 10
  thermo_every: 5
  cutoff: 2.5
td:
  cell_size: 1.0
  zones_total: 2
  zone_cells_w: 1
  zone_cells_s: 1
  traversal: snake
"""
    path = tmp_path / "default_nve.yaml"
    path.write_text(cfg_text, encoding="utf-8")
    cfg = load_config(str(path))
    assert cfg.ensemble.kind == "nve"
    assert cfg.ensemble.thermostat is None
    assert cfg.ensemble.barostat is None


def test_load_config_rejects_invalid_ensemble_contract(tmp_path):
    cfg_text = """
system:
  n_atoms: 8
  mass: 1.0
  box: 10.0
  temperature: 1.0
  seed: 1
potential:
  kind: lj
  params:
    epsilon: 1.0
    sigma: 1.0
run:
  dt: 0.005
  n_steps: 10
  thermo_every: 5
  cutoff: 2.5
td:
  cell_size: 1.0
  zones_total: 2
  zone_cells_w: 1
  zone_cells_s: 1
  traversal: snake
ensemble:
  kind: npt
  thermostat:
    kind: langevin
    params:
      tau: 0.2
      t_target: 300.0
"""
    path = tmp_path / "bad_ensemble.yaml"
    path.write_text(cfg_text, encoding="utf-8")
    with pytest.raises(ValueError, match="requires ensemble.barostat"):
        load_config(str(path))
