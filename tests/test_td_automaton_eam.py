from __future__ import annotations

import numpy as np

from tdmd.potentials import make_potential
from tdmd.serial import run_serial
from tdmd.td_automaton import ZoneRuntime, TDAutomaton1W
from tdmd.zone_bins_localz import PersistentZoneLocalZBinsCache
from tdmd.zones import ZoneType


def test_td_automaton_compute_step_uses_eam_many_body_forces():
    box = 10.0
    cutoff = 5.0
    dt = 1e-3
    mass = 26.9815386
    atom_types = np.array([1, 2], dtype=np.int32)
    pot = make_potential(
        "eam/alloy",
        {"file": "examples/interop/eam_al_ni_demo.setfl", "elements": ["Al", "Ni"]},
    )
    r0 = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    v0 = np.zeros_like(r0)

    # serial reference (one global step)
    r_ref = r0.copy()
    v_ref = v0.copy()
    run_serial(r_ref, v_ref, mass, box, pot, dt, cutoff, n_steps=1, atom_types=atom_types)

    # automaton step on one zone
    r = r0.copy()
    v = v0.copy()
    zones = [
        ZoneRuntime(
            zid=0,
            z0=0.0,
            z1=float(box),
            ztype=ZoneType.D,
            atom_ids=np.array([0, 1], dtype=np.int32),
            step_id=0,
        )
    ]
    zones[0].n_cells = 1  # compatibility with zones_overlapping_interval helper
    autom = TDAutomaton1W(
        zones_runtime=zones,
        box=box,
        cutoff=cutoff,
        bins_cache=PersistentZoneLocalZBinsCache(),
        traversal_order=[0],
        formal_core=True,
    )
    zid = autom.start_compute(r=r, rc=cutoff, skin_global=0.0, step=1, verlet_k_steps=1)
    assert zid == 0
    done = autom.compute_step_for_work_zone(
        r=r,
        v=v,
        mass=mass,
        dt=dt,
        potential=pot,
        cutoff=cutoff,
        rc=cutoff,
        skin_global=0.0,
        step=1,
        verlet_k_steps=1,
        atom_types=atom_types,
        enable_step_id=True,
    )
    assert done == 0
    assert np.allclose(r, r_ref, atol=1e-12, rtol=1e-12)
    assert np.allclose(v, v_ref, atol=1e-12, rtol=1e-12)
