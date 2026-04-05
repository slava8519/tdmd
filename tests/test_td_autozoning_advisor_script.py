from __future__ import annotations

import json
import subprocess
import sys


def test_td_autozoning_advisor_script_writes_artifacts(tmp_path):
    out_csv = tmp_path / "td_autozoning_advisor_gpu.csv"
    out_md = tmp_path / "td_autozoning_advisor_gpu.md"
    out_json = tmp_path / "td_autozoning_advisor_gpu.summary.json"
    cmd = [
        sys.executable,
        "scripts/bench_td_autozoning_advisor.py",
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
        "--zone-totals",
        "2",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out_csv.is_file()
    assert out_md.is_file()
    assert out_json.is_file()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert bool(data.get("ok_all")) is True
    assert int(data.get("total", 0)) == 1
    assert "resources" in data
    assert "recommendation" in data
    assert "breakdown" in data
    assert dict(data.get("recommendation", {})).get("recommended_td_layout", {}).get("zones_total") == 2
    assert dict(data.get("breakdown", {})).get("force_scope_contract", {}).get("version") == "pr_za01_v1"
    assert int(
        dict(dict(data.get("breakdown", {})).get("by_case", {}).get("time_gpu", {}))
        .get("breakdown", {})
        .get("target_local_force_calls", 0)
    ) > 0

    report = out_md.read_text(encoding="utf-8")
    assert "TD Auto-Zoning Advisor" in report
    assert "Resource Snapshot" in report
    assert "Candidate Layouts" in report
    assert "Recommendation" in report
    assert "Breakdown Evidence" in report
    assert "TD Auto-Zoning Advisor" in proc.stdout
