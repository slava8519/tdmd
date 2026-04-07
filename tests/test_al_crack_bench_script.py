from __future__ import annotations

import json
import subprocess
import sys

from tdmd.io import load_task


def test_generate_al_crack_task_writes_pure_al_eam_block(tmp_path):
    out = tmp_path / "task_al_crack_small.yaml"
    cmd = [
        sys.executable,
        "scripts/generate_al_crack_task.py",
        "--out",
        str(out),
        "--target-atoms",
        "2048",
        "--box",
        "40.0",
        "--steps",
        "5",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "kind: eam/alloy" in text
    assert "elements: [Al]" in text
    assert "file: examples/potentials/eam_alloy/Al_zhou.eam.alloy" in text
    task = load_task(str(out))
    assert len(task.atoms) == 2048


def test_bench_al_crack_compare_script_writes_artifacts(tmp_path):
    out_csv = tmp_path / "al_crack_compare.csv"
    out_md = tmp_path / "al_crack_compare.md"
    out_json = tmp_path / "al_crack_compare.summary.json"
    telemetry_dir = tmp_path / "telemetry"
    task_out = tmp_path / "task_al_crack_small.yaml"
    cmd = [
        sys.executable,
        "scripts/bench_al_crack_compare.py",
        "--task-out",
        str(task_out),
        "--out",
        str(out_csv),
        "--md",
        str(out_md),
        "--json",
        str(out_json),
        "--target-atoms",
        "512",
        "--box",
        "40.0",
        "--steps",
        "1",
        "--device",
        "cpu",
        "--requested-zones",
        "2",
        "--cell-size",
        "2.0",
        "--timeout-sec",
        "60",
        "--requested-space-timeout-sec",
        "20",
        "--compare-space-timeout-sec",
        "20",
        "--compare-time-timeout-sec",
        "20",
        "--telemetry-dir",
        str(telemetry_dir),
        "--telemetry-every",
        "1",
        "--telemetry-heartbeat-sec",
        "0.1",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out_csv.is_file()
    assert out_md.is_file()
    assert out_json.is_file()
    assert task_out.is_file()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert bool(data.get("ok_all")) is True
    assert bool(data.get("generated_task")) is True
    assert bool(data.get("requested_td_preflight", {}).get("preflight_ok")) is True
    assert str(data.get("wavefront_contract_version", "")) == "pr_sw01_v1"
    assert bool(dict(data.get("requested_td_wavefront", {})).get("layout_valid")) is True
    assert int(data.get("strict_valid_common_zones_total", 0)) == 2
    assert (
        int(dict(data.get("strict_valid_common_td_wavefront", {})).get("first_wave_size", 0)) >= 1
    )
    assert len(list(data.get("rows", []))) == 2
    for row in data.get("rows", []):
        assert row["effective_device"] == "cpu"
        assert row["telemetry_path"]
        assert row["telemetry_summary_path"]
        assert int(row["last_step_observed"]) == 1

    report = out_md.read_text(encoding="utf-8")
    assert "Al Crack Decomposition Benchmark" in report
    assert "Strict-Valid Wavefront Contract" in report
    assert "strict_valid_td_speedup_vs_space" in report
    assert "max_valid_td_zones_total" in report
    assert "Al Crack Decomposition Benchmark" in proc.stdout
