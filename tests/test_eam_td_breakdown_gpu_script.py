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

    report = out_md.read_text(encoding="utf-8")
    assert "EAM TD GPU Breakdown" in report
    assert "forces_full_total_sec" in report
    assert "device_sync_sec" in report
    assert "zone_assign_sec" in report
    assert "EAM TD GPU Breakdown" in proc.stdout
