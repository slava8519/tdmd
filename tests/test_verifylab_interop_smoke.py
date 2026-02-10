from __future__ import annotations

from tdmd.config import load_config
from tdmd.verify_lab import sweep_verify_task


def test_verifylab_interop_smoke():
    cfg = load_config("examples/td_1d_morse.yaml")
    rows = sweep_verify_task(
        "examples/interop/task.yaml",
        cfg.td,
        steps=2,
        every=1,
        zones_total_list=[2],
        use_verlet_list=[False],
        verlet_k_steps_list=[5],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
    )
    assert rows, "no rows produced"
    bad = [r for r in rows if not r.ok]
    assert not bad, f"interop_smoke failed: {bad[:3]}"


def test_verifylab_interop_smoke_binary_alloy_task():
    cfg = load_config("examples/td_1d_morse.yaml")
    rows = sweep_verify_task(
        "examples/interop/task_alloy_pair.yaml",
        cfg.td,
        steps=2,
        every=1,
        zones_total_list=[2],
        use_verlet_list=[False],
        verlet_k_steps_list=[5],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
    )
    assert rows, "no rows produced"
    bad = [r for r in rows if not r.ok]
    assert not bad, f"interop_smoke alloy task failed: {bad[:3]}"


def test_verifylab_interop_smoke_table_task():
    cfg = load_config("examples/td_1d_morse.yaml")
    rows = sweep_verify_task(
        "examples/interop/task_table.yaml",
        cfg.td,
        steps=2,
        every=1,
        zones_total_list=[2],
        use_verlet_list=[False],
        verlet_k_steps_list=[5],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
    )
    assert rows, "no rows produced"
    bad = [r for r in rows if not r.ok]
    assert not bad, f"interop_smoke table task failed: {bad[:3]}"


def test_verifylab_interop_eam_task_sync_mode():
    cfg = load_config("examples/td_1d_morse.yaml")
    rows = sweep_verify_task(
        "examples/interop/task_eam_alloy.yaml",
        cfg.td,
        steps=2,
        every=1,
        zones_total_list=[2],
        use_verlet_list=[False],
        verlet_k_steps_list=[5],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
        sync_mode=True,
    )
    assert rows, "no rows produced"
    bad = [r for r in rows if not r.ok]
    assert not bad, f"interop_smoke EAM sync task failed: {bad[:3]}"
