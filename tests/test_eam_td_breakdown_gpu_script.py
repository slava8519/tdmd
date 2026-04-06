from __future__ import annotations

import json
import subprocess
import sys


def test_eam_td_breakdown_gpu_script_writes_artifacts(tmp_path):
    out_csv = tmp_path / "eam_td_breakdown_gpu.csv"
    out_md = tmp_path / "eam_td_breakdown_gpu.md"
    out_json = tmp_path / "eam_td_breakdown_gpu.summary.json"
    cmd = [
        sys.executable,
        "scripts/bench_eam_td_breakdown_gpu.py",
        "--out",
        str(out_csv),
        "--md",
        str(out_md),
        "--json",
        str(out_json),
        "--n-atoms",
        "256",
        "--steps",
        "1",
        "--repeats",
        "1",
        "--warmup",
        "0",
        "--zones-total",
        "2",
        "--zones-nx",
        "2",
        "--zones-ny",
        "1",
        "--zones-nz",
        "1",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out_csv.is_file()
    assert out_md.is_file()
    assert out_json.is_file()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert bool(data.get("ok_all")) is True
    assert int(data.get("total", 0)) == 2
    assert list(sorted(data.get("by_case", {}).keys())) == ["space_gpu", "time_gpu"]
    assert data.get("force_scope_contract", {}).get("version") == "pr_mb03_v1"
    time_breakdown = data["by_case"]["time_gpu"]["breakdown"]
    assert time_breakdown["many_body_evaluation_scope"] == "target_local"
    assert time_breakdown["many_body_consumption_scope"] == "target_ids"
    assert int(time_breakdown["many_body_target_local_available"]) == 1
    assert int(time_breakdown["target_local_force_calls"]) > 0

    report = out_md.read_text(encoding="utf-8")
    assert "EAM TD GPU Breakdown" in report
    assert "many_body_eval_scope" in report
    assert "target_local_force_calls" in report
    assert "forces_full_total_sec" in report
    assert "device_sync_sec" in report
    assert "zone_assign_sec" in report
    assert "EAM TD GPU Breakdown" in proc.stdout
