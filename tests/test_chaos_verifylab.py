from tdmd.config import load_config
from tdmd.potentials import make_potential
from tdmd.verify_lab import summarize, sweep_verify2


def test_verifylab_with_chaos_smoke():
    cfg = load_config("examples/td_1d_morse.yaml")
    pot = make_potential(cfg.potential.kind, cfg.potential.params)
    rows = sweep_verify2(
        cfg,
        pot,
        steps=40,
        every=10,
        zones_total_list=[cfg.td.zones_total],
        use_verlet_list=[cfg.td.use_verlet],
        verlet_k_steps_list=[cfg.td.verlet_k_steps],
        chaos_mode_list=[True],
        chaos_delay_prob_list=[0.05],
        chaos_seed=123,
        # relaxed tolerances, chaos can reorder but should remain close
        tol=dict(tol_dr=5e-5, tol_dv=5e-5, tol_dE=5e-4, tol_dT=5e-4, tol_dP=5e-3),
    )
    summ = summarize(rows)
    assert summ["total"] > 0
