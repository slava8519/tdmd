from __future__ import annotations

from tdmd.config import load_config
from tdmd.verify_lab import sweep_verify_task


def test_sweep_verify_task_accepts_ml_reference_task():
    cfg = load_config("examples/td_1d_morse.yaml")
    rows = sweep_verify_task(
        "examples/interop/task_ml_reference.yaml",
        cfg.td,
        steps=1,
        every=1,
        zones_total_list=[1],
        use_verlet_list=[False],
        verlet_k_steps_list=[1],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
        sync_mode=True,
        device="cpu",
        strict_min_zone_width=True,
    )
    assert rows
    assert all(bool(row.ok) for row in rows)
