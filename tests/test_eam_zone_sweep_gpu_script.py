from __future__ import annotations

import json
import subprocess
import sys


def test_eam_zone_sweep_gpu_script_writes_artifacts(tmp_path):
    out_csv = tmp_path / "eam_zone_sweep_gpu.csv"
    out_md = tmp_path / "eam_zone_sweep_gpu.md"
    out_json = tmp_path / "eam_zone_sweep_gpu.summary.json"
    cmd = [
        sys.executable,
        "scripts/bench_eam_zone_sweep_gpu.py",
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
        "--layouts",
        "1:1x1x1,2:2x1x1",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out_csv.is_file()
    assert out_md.is_file()
    assert out_json.is_file()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert bool(data.get("ok_all")) is True
    assert int(data.get("total", 0)) == 2
    assert list(data.get("selected_layouts", [])) == ["1:1x1x1", "2:2x1x1"]
    assert len(list(data.get("rows", []))) == 2

    report = out_md.read_text(encoding="utf-8")
    assert "EAM Alloy GPU Zone Sweep" in report
    assert "td_speedup_vs_space" in report
    assert "Best Observed Layouts" in report
    assert "EAM Alloy GPU Zone Sweep" in proc.stdout
