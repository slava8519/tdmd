from __future__ import annotations

import numpy as np

from tdmd.io import load_task, task_to_arrays, validate_task_for_run
from tdmd.potentials import make_potential
from tdmd.serial import run_serial
from tdmd.td_local import run_td_local


def _run_serial_vs_td_local(
    task_path: str, *, sync_mode: bool, zones_total: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    task = load_task(task_path)
    arr = task_to_arrays(task)
    masses = validate_task_for_run(task, allowed_potential_kinds=("eam/alloy",))
    pot = make_potential(task.potential.kind, task.potential.params)
    box = float(task.box.x)
    dt = float(task.dt)
    cutoff = float(task.cutoff)

    r_serial = arr.r.copy()
    v_serial = arr.v.copy()
    run_serial(
        r_serial,
        v_serial,
        masses,
        box,
        pot,
        dt,
        cutoff,
        n_steps=int(task.steps),
        atom_types=arr.atom_types,
    )

    r_td = arr.r.copy()
    v_td = arr.v.copy()
    run_td_local(
        r_td,
        v_td,
        masses,
        box,
        pot,
        dt,
        cutoff,
        n_steps=int(task.steps),
        atom_types=arr.atom_types,
        cell_size=max(1.0, cutoff / 2.0),
        zones_total=int(zones_total),
        zone_cells_w=1,
        zone_cells_s=1,
        traversal="forward",
        use_verlet=False,
        decomposition="1d",
        sync_mode=bool(sync_mode),
    )
    return r_serial, v_serial, r_td, v_td


def test_td_local_eam_sync_mode_matches_serial():
    r_s, v_s, r_t, v_t = _run_serial_vs_td_local(
        "examples/interop/task_eam_alloy.yaml",
        sync_mode=True,
        zones_total=2,
    )
    assert np.allclose(r_s, r_t, atol=1e-12, rtol=1e-12)
    assert np.allclose(v_s, v_t, atol=1e-12, rtol=1e-12)


def test_td_local_eam_async_single_zone_matches_serial():
    r_s, v_s, r_t, v_t = _run_serial_vs_td_local(
        "examples/interop/task_eam_al.yaml",
        sync_mode=False,
        zones_total=1,
    )
    assert np.allclose(r_s, r_t, atol=1e-12, rtol=1e-12)
    assert np.allclose(v_s, v_t, atol=1e-12, rtol=1e-12)
