from __future__ import annotations

from tdmd.config import load_config
from tdmd.potentials import make_potential
from tdmd.verify_v2 import run_verify_config


def test_liveness_chaos_no_stall_cfg_system(tmp_path):
    # Uses td_local chaos shuffling + A4b priority scheduling; test is about completion/progress, not strict equality.
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
        zones_total=int(cfg.td.zones_total),
        zone_cells_w=int(cfg.td.zone_cells_w),
        zone_cells_s=int(cfg.td.zone_cells_s),
        zone_cells_pattern=cfg.td.zone_cells_pattern,
        traversal=str(cfg.td.traversal),
        buffer_k=float(cfg.td.buffer_k),
        skin_from_buffer=bool(cfg.td.skin_from_buffer),
        use_verlet=bool(cfg.td.use_verlet),
        verlet_k_steps=int(cfg.td.verlet_k_steps),
        steps=40,
        observer_every=10,
        # very loose tol: this is a liveness test, not an equivalence test
        tol_dr=1e9, tol_dv=1e9, tol_dE=1e9, tol_dT=1e9, tol_dP=1e9,
        decomposition=str(getattr(cfg.td, "decomposition", "1d")),
        zones_nx=int(getattr(cfg.td, "zones_nx", 1)),
        zones_ny=int(getattr(cfg.td, "zones_ny", 1)),
        zones_nz=int(getattr(cfg.td, "zones_nz", 1)),
        chaos_mode=True,
        chaos_seed=123,
        chaos_delay_prob=0.5,
    )

    assert res.steps == 40
    # progress_epoch should be present in details (td_local automaton diag)
    pe = res.details.get("progress_epoch", None)
    assert pe is None or pe > 0
