from __future__ import annotations

import numpy as np

from tdmd.io.trajectory import TrajectoryWriter
from tdmd.potentials import make_potential
from tdmd.serial import run_serial
from tdmd.td_local import run_td_local
from tdmd.td_trace import TDTraceLogger


def test_outputs_smoke(tmp_path):
    n = 4
    box = 10.0
    mass = 1.0
    dt = 0.005
    cutoff = 2.5
    pot = make_potential("lj", {})

    r = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]], dtype=float)
    v = np.zeros_like(r)

    traj_path = tmp_path / "traj.lammpstrj"
    writer = TrajectoryWriter(
        str(traj_path),
        box=(box, box, box),
        pbc=(True, True, True),
        atom_ids=np.arange(n, dtype=np.int32) + 1,
        atom_types=np.ones(n, dtype=np.int32),
    )

    def obs(step, r0, v0):
        writer.write(step, r0, v0)

    run_serial(
        r.copy(),
        v.copy(),
        mass,
        box,
        pot,
        dt,
        cutoff,
        n_steps=2,
        thermo_every=0,
        observer=obs,
        observer_every=1,
    )
    writer.close()
    text = traj_path.read_text(encoding="utf-8")
    assert "ITEM: TIMESTEP" in text

    trace_path = tmp_path / "td_trace.csv"
    trace = TDTraceLogger(str(trace_path), rank=0, enabled=True)
    run_td_local(
        r.copy(),
        v.copy(),
        mass,
        box,
        pot,
        dt,
        cutoff,
        n_steps=2,
        observer=None,
        observer_every=0,
        trace=trace,
        cell_size=2.5,
        zones_total=2,
        zone_cells_w=1,
        zone_cells_s=1,
        traversal="forward",
        buffer_k=1.2,
        skin_from_buffer=True,
        use_verlet=False,
        verlet_k_steps=1,
        decomposition="1d",
        sync_mode=False,
    )
    trace.close()
    lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2
