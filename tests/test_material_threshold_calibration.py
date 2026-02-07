from __future__ import annotations

import json
import subprocess
import sys


def test_calibrate_material_thresholds_script(tmp_path):
    out_json = tmp_path / "policy.json"
    out_md = tmp_path / "policy.md"
    cmd = [
        sys.executable,
        "scripts/calibrate_material_thresholds.py",
        "--fixture",
        "examples/interop/materials_parity_suite_v2.json",
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert int(data["policy_version"]) == 2
    thr = dict(data.get("thresholds", {}))
    assert "eos_energy_abs" in thr
    assert "thermo_energy_abs" in thr
    assert "transport_verify_abs" in thr
