from __future__ import annotations

import sys

import numpy as np
import pytest

import tdmd.main as tdmd_main


def test_run_task_td_full_mpi_passes_nonuniform_mass_array(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_run_td_full_mpi_1d(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(tdmd_main, "run_td_full_mpi_1d", _fake_run_td_full_mpi_1d)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tdmd",
            "run",
            "examples/td_1d_morse.yaml",
            "--task",
            "examples/interop/task_eam_alloy.yaml",
            "--mode",
            "td_full_mpi",
        ],
    )

    tdmd_main.main()

    assert "mass" in captured
    mass = captured["mass"]
    assert isinstance(mass, np.ndarray)
    assert mass.shape == (2,)
    assert mass.tolist() == pytest.approx([26.9815386, 58.6934])


def test_main_passes_config_ensemble_to_serial_runtime(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def _fake_run_serial(*args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(tdmd_main, "run_serial", _fake_run_serial)

    cfg_path = tmp_path / "cfg_nvt.yaml"
    cfg_path.write_text(
        """
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
  n_steps: 2
  thermo_every: 1
  cutoff: 2.5
td:
  cell_size: 1.0
  zones_total: 2
  zone_cells_w: 1
  zone_cells_s: 1
  traversal: snake
ensemble:
  kind: nvt
  thermostat:
    kind: berendsen
    params:
      tau: 0.1
      t_target: 300.0
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tdmd",
            "run",
            str(cfg_path),
            "--mode",
            "serial",
        ],
    )
    tdmd_main.main()
    assert captured.get("ensemble_kind") == "nvt"
    thermostat = captured.get("thermostat")
    assert thermostat is not None
    assert getattr(thermostat, "kind", "") == "berendsen"
