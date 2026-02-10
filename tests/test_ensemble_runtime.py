from __future__ import annotations

import numpy as np
import pytest

import tdmd.td_full_mpi as td_full_mpi
from tdmd.io import export_lammps_in, load_task, task_to_arrays, validate_task_for_run
from tdmd.potentials import make_potential
from tdmd.serial import run_serial
from tdmd.state import kinetic_energy, temperature_from_ke
from tdmd.verify_v2 import run_verify_task


def _tiny_state() -> tuple[np.ndarray, np.ndarray]:
    r = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
        ],
        dtype=float,
    )
    v = np.array(
        [
            [0.3, 0.1, -0.2],
            [0.0, -0.2, 0.1],
            [-0.1, 0.2, 0.0],
            [0.2, -0.1, 0.1],
        ],
        dtype=float,
    )
    return r, v


def test_serial_nvt_berendsen_moves_temperature_towards_target():
    r, v = _tiny_state()
    mass = 1.0
    box = 10.0
    pot = make_potential("table", {"file": "examples/interop/table_zero.table", "keyword": "ZERO"})

    t0 = temperature_from_ke(kinetic_energy(v, mass), int(v.shape[0]))
    run_serial(
        r,
        v,
        mass,
        box,
        pot,
        dt=0.01,
        cutoff=2.5,
        n_steps=80,
        ensemble_kind="nvt",
        thermostat={"kind": "berendsen", "params": {"t_target": 0.5, "tau": 0.1}},
    )
    t1 = temperature_from_ke(kinetic_energy(v, mass), int(v.shape[0]))

    assert abs(t1 - 0.5) < abs(t0 - 0.5)
    assert abs(t1 - 0.5) < 0.25


def test_serial_npt_berendsen_changes_box():
    r, v = _tiny_state()
    mass = 1.0
    box0 = 10.0
    pot = make_potential("table", {"file": "examples/interop/table_zero.table", "keyword": "ZERO"})
    boxes: list[float] = []

    def _obs(step, rr, vv, box_cur=None):
        boxes.append(float(box_cur if box_cur is not None else box0))

    run_serial(
        r,
        v,
        mass,
        box0,
        pot,
        dt=0.01,
        cutoff=2.5,
        n_steps=40,
        observer=_obs,
        observer_every=1,
        ensemble_kind="npt",
        thermostat={"kind": "berendsen", "params": {"t_target": 1.0, "tau": 0.1}},
        barostat={
            "kind": "berendsen",
            "params": {
                "p_target": 0.0,
                "tau": 0.5,
                "compressibility": 0.02,
                "scale_velocities": True,
            },
        },
    )

    assert len(boxes) > 2
    assert boxes[-1] != pytest.approx(boxes[0])


def test_verify_task_nvt_sync_serial_tdlocal_ok():
    task = load_task("examples/interop/task_nvt.yaml")
    masses = validate_task_for_run(
        task,
        allowed_potential_kinds=("lj", "morse", "table", "eam/alloy"),
        allowed_ensemble_kinds=("nve", "nvt", "npt"),
    )
    arr = task_to_arrays(task)
    pot = make_potential(task.potential.kind, task.potential.params)

    res = run_verify_task(
        potential=pot,
        r0=arr.r,
        v0=arr.v,
        box=float(task.box.x),
        mass=masses,
        dt=float(task.dt),
        cutoff=float(task.cutoff),
        atom_types=arr.atom_types,
        cell_size=1.0,
        zones_total=1,
        zone_cells_w=1,
        zone_cells_s=1,
        zone_cells_pattern=None,
        traversal="forward",
        buffer_k=1.2,
        skin_from_buffer=True,
        use_verlet=False,
        verlet_k_steps=1,
        steps=20,
        observer_every=5,
        tol_dr=1e-5,
        tol_dv=1e-5,
        tol_dE=1e-4,
        tol_dT=1e-4,
        tol_dP=1e-3,
        sync_mode=True,
        ensemble_kind=task.ensemble.kind,
        thermostat=task.ensemble.thermostat,
        barostat=task.ensemble.barostat,
    )
    assert res.ok


def test_verify_task_npt_sync_serial_tdlocal_ok():
    task = load_task("examples/interop/task_npt.yaml")
    masses = validate_task_for_run(
        task,
        allowed_potential_kinds=("lj", "morse", "table", "eam/alloy"),
        allowed_ensemble_kinds=("nve", "nvt", "npt"),
    )
    arr = task_to_arrays(task)
    pot = make_potential(task.potential.kind, task.potential.params)

    res = run_verify_task(
        potential=pot,
        r0=arr.r,
        v0=arr.v,
        box=float(task.box.x),
        mass=masses,
        dt=float(task.dt),
        cutoff=float(task.cutoff),
        atom_types=arr.atom_types,
        cell_size=1.0,
        zones_total=1,
        zone_cells_w=1,
        zone_cells_s=1,
        zone_cells_pattern=None,
        traversal="forward",
        buffer_k=1.2,
        skin_from_buffer=True,
        use_verlet=False,
        verlet_k_steps=1,
        steps=20,
        observer_every=5,
        tol_dr=1e-5,
        tol_dv=1e-5,
        tol_dE=1e-4,
        tol_dT=1e-4,
        tol_dP=1e-3,
        sync_mode=True,
        ensemble_kind=task.ensemble.kind,
        thermostat=task.ensemble.thermostat,
        barostat=task.ensemble.barostat,
    )
    assert res.ok


def test_lammps_in_export_emits_fix_for_nvt_and_npt(tmp_path):
    t_nvt = load_task("examples/interop/task_nvt.yaml")
    nvt_in = tmp_path / "nvt.in"
    export_lammps_in(t_nvt, str(nvt_in), data_filename="data.lammps")
    txt_nvt = nvt_in.read_text(encoding="utf-8")
    assert "fix tdmd_int all nvt" in txt_nvt

    t_npt = load_task("examples/interop/task_npt.yaml")
    npt_in = tmp_path / "npt.in"
    export_lammps_in(t_npt, str(npt_in), data_filename="data.lammps")
    txt_npt = npt_in.read_text(encoding="utf-8")
    assert "fix tdmd_int all npt" in txt_npt


def test_td_full_mpi_rejects_npt_for_multi_rank(monkeypatch):
    class _FakeComm:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 2

    class _FakeMPI:
        COMM_WORLD = _FakeComm()

        @staticmethod
        def Is_initialized():
            return True

        @staticmethod
        def Init():
            return None

    monkeypatch.setattr(td_full_mpi, "MPI", _FakeMPI)
    r, v = _tiny_state()
    pot = make_potential("lj", {"epsilon": 1.0, "sigma": 1.0})
    with pytest.raises(ValueError, match="single-rank"):
        td_full_mpi.run_td_full_mpi_1d(
            r=r,
            v=v,
            mass=1.0,
            box=10.0,
            potential=pot,
            dt=0.005,
            cutoff=2.5,
            n_steps=1,
            thermo_every=0,
            cell_size=1.0,
            zones_total=1,
            zone_cells_w=1,
            zone_cells_s=1,
            zone_cells_pattern=None,
            traversal="forward",
            fast_sync=False,
            strict_fast_sync=False,
            startup_mode="scatter_zones",
            warmup_steps=0,
            warmup_compute=False,
            buffer_k=1.2,
            use_verlet=False,
            verlet_k_steps=1,
            skin_from_buffer=True,
            ensemble_kind="npt",
            thermostat={"kind": "berendsen", "params": {"t_target": 1.0, "tau": 0.1}},
            barostat={"kind": "berendsen", "params": {"p_target": 0.0, "tau": 1.0}},
        )
