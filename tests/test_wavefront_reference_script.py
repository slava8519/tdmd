from __future__ import annotations

import json
import subprocess
import sys


def test_wavefront_reference_script_writes_artifacts(tmp_path):
    out_csv = tmp_path / "wavefront_reference_equivalence.csv"
    out_md = tmp_path / "wavefront_reference_equivalence.md"
    out_json = tmp_path / "wavefront_reference_equivalence.summary.json"
    cmd = [
        sys.executable,
        "scripts/bench_wavefront_reference_equivalence.py",
        "--out",
        str(out_csv),
        "--md",
        str(out_md),
        "--json",
        str(out_json),
        "--morse-steps",
        "2",
        "--eam-alloy-n-atoms",
        "256",
        "--eam-alloy-steps",
        "1",
        "--eam-alloy-zones-total",
        "2",
        "--eam-alloy-cell-size",
        "2.7",
        "--allow-no-multi-zone-wave",
        "--strict",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out_csv.is_file()
    assert out_md.is_file()
    assert out_json.is_file()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert bool(data.get("ok_all")) is True
    assert int(data.get("total", 0)) == 2
    assert "morse_cfg_pair" in dict(data.get("by_case", {}))
    assert "eam_alloy_dense" in dict(data.get("by_case", {}))
    assert "Wavefront Reference Equivalence" in str(data.get("report_markdown", ""))

    report = out_md.read_text(encoding="utf-8")
    assert "Wavefront Reference Equivalence" in report
    assert "morse_cfg_pair" in report
    assert "eam_alloy_dense" in report
    assert "Wavefront Reference Equivalence" in proc.stdout
