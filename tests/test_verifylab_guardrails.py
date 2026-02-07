from __future__ import annotations

import pytest

from tdmd.config import load_config
from tdmd.verify_lab import sweep_verify_task


def test_task_verify_guardrail_can_fail_on_invalid_zone_width_strict():
    cfg = load_config("examples/td_1d_morse.yaml")
    with pytest.raises(ValueError, match="min_zone_width"):
        sweep_verify_task(
            "examples/interop/task.yaml",
            cfg.td,
            steps=2,
            every=1,
            zones_total_list=[4],
            use_verlet_list=[False],
            verlet_k_steps_list=[10],
            chaos_mode_list=[False],
            chaos_delay_prob_list=[0.0],
            strict_min_zone_width=True,
        )


def test_task_verify_guardrail_keeps_diagnostic_mode():
    cfg = load_config("examples/td_1d_morse.yaml")
    rows = sweep_verify_task(
        "examples/interop/task.yaml",
        cfg.td,
        steps=2,
        every=1,
        zones_total_list=[4],
        use_verlet_list=[False],
        verlet_k_steps_list=[10],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
        strict_min_zone_width=False,
    )
    assert rows, "expected diagnostic rows"
