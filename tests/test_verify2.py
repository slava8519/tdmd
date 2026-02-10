import numpy as np

from tdmd.config import load_config
from tdmd.potentials import make_potential
from tdmd.verify_v2 import run_verify_config


def test_verify2_cfg_system():
    # Use the example config as a baseline (cfg_system)
    cfg = load_config("examples/td_1d_morse.yaml")
    pot = make_potential(cfg.potential.kind, cfg.potential.params)

    res = run_verify_config(
        potential=pot,
        n_atoms=int(cfg.system.n_atoms),
        box=float(cfg.system.box),
        temperature=float(cfg.system.temperature),
        seed=int(cfg.system.seed),
        mass=float(cfg.system.mass),
        dt=float(cfg.run.dt),
        cutoff=float(cfg.run.cutoff),
        cell_size=float(cfg.td.cell_size),
        zones_total=4,
        zone_cells_w=int(cfg.td.zone_cells_w),
        zone_cells_s=int(cfg.td.zone_cells_s),
        zone_cells_pattern=cfg.td.zone_cells_pattern,
        traversal=str(cfg.td.traversal),
        buffer_k=float(cfg.td.buffer_k),
        skin_from_buffer=bool(cfg.td.skin_from_buffer),
        use_verlet=False,
        verlet_k_steps=10,
        steps=2,
        observer_every=1,
        # CI: single-step equivalence check on cfg_system
        tol_dr=1e-5,
        tol_dv=3e-5,
        tol_dE=2.5e-4,
        tol_dT=1e-4,
        tol_dP=1e-3,
        decomposition=str(getattr(cfg.td, "decomposition", "1d")),
        zones_nx=int(getattr(cfg.td, "zones_nx", 1)),
        zones_ny=int(getattr(cfg.td, "zones_ny", 1)),
        zones_nz=int(getattr(cfg.td, "zones_nz", 1)),
        chaos_mode=False,
        chaos_seed=int(cfg.td.chaos_seed),
        chaos_delay_prob=0.0,
    )
    assert res.ok, (res.case, res.max_dr, res.max_dv, res.max_dE, res.max_dT, res.max_dP)


def test_verify2_cfg_system_sync_mode():
    cfg = load_config("examples/td_1d_morse.yaml")
    pot = make_potential(cfg.potential.kind, cfg.potential.params)

    res = run_verify_config(
        potential=pot,
        n_atoms=int(cfg.system.n_atoms),
        box=float(cfg.system.box),
        temperature=float(cfg.system.temperature),
        seed=int(cfg.system.seed),
        mass=float(cfg.system.mass),
        dt=float(cfg.run.dt),
        cutoff=float(cfg.run.cutoff),
        cell_size=float(cfg.td.cell_size),
        zones_total=4,
        zone_cells_w=int(cfg.td.zone_cells_w),
        zone_cells_s=int(cfg.td.zone_cells_s),
        zone_cells_pattern=cfg.td.zone_cells_pattern,
        traversal=str(cfg.td.traversal),
        buffer_k=float(cfg.td.buffer_k),
        skin_from_buffer=bool(cfg.td.skin_from_buffer),
        use_verlet=False,
        verlet_k_steps=10,
        steps=10,
        observer_every=2,
        tol_dr=1e-6,
        tol_dv=1e-6,
        tol_dE=1e-6,
        tol_dT=1e-6,
        tol_dP=1e-6,
        decomposition=str(getattr(cfg.td, "decomposition", "1d")),
        zones_nx=int(getattr(cfg.td, "zones_nx", 1)),
        zones_ny=int(getattr(cfg.td, "zones_ny", 1)),
        zones_nz=int(getattr(cfg.td, "zones_nz", 1)),
        chaos_mode=False,
        chaos_seed=int(cfg.td.chaos_seed),
        chaos_delay_prob=0.0,
        sync_mode=True,
    )
    assert res.ok, (res.case, res.max_dr, res.max_dv, res.max_dE, res.max_dT, res.max_dP)
