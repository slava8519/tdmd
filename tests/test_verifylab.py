from tdmd.config import load_config
from tdmd.potentials import make_potential
from tdmd.verify_lab import summarize, sweep_verify2


def test_verifylab_small_sweep():
    cfg = load_config("examples/td_1d_morse.yaml")
    pot = make_potential(cfg.potential.kind, cfg.potential.params)
    rows = sweep_verify2(
        cfg,
        pot,
        steps=30,
        every=10,
        zones_total_list=[cfg.td.zones_total, max(1, cfg.td.zones_total // 2)],
        use_verlet_list=[cfg.td.use_verlet],
        verlet_k_steps_list=[cfg.td.verlet_k_steps],
    )
    summ = summarize(rows)
    assert summ["total"] > 0
