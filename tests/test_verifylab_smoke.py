from __future__ import annotations

from tdmd.config import load_config
from tdmd.potentials import make_potential
from tdmd.verify_lab import sweep_verify2


def test_verifylab_smoke():
    cfg = load_config("examples/td_1d_morse.yaml")
    pot = make_potential(cfg.potential.kind, cfg.potential.params)
    tol = dict(tol_dr=1e-5, tol_dv=3e-5, tol_dE=2.5e-4, tol_dT=1e-4, tol_dP=1e-3)
    rows = sweep_verify2(
        cfg,
        pot,
        steps=2,
        every=1,
        zones_total_list=[4],
        use_verlet_list=[False],
        verlet_k_steps_list=[10],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
        tol=tol,
    )
    assert rows, "no rows produced"
    bad = []
    for r in rows:
        violations = list(r.details.get("violations", []))
        inv = r.details.get("invariants", {})
        for key in ("hG3", "hV3", "tG3"):
            if int(inv.get(key, 0)) != 0:
                violations.append(f"{key}={int(inv.get(key, 0))}")
        if violations:
            tol_used = r.details.get("tol", {})
            bad.append(
                f"case={r.case} zt={r.zones_total} verlet={r.use_verlet} chaos={r.chaos_mode} "
                f"dr={r.max_dr:.3e} dv={r.max_dv:.3e} dE={r.max_dE:.3e} dT={r.max_dT:.3e} dP={r.max_dP:.3e} "
                f"tol={tol_used} violations={violations}"
            )
    assert not bad, f"verifylab smoke failed: {bad[:3]}"
