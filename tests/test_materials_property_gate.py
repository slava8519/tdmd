from __future__ import annotations

import json
import subprocess
import sys


def test_materials_property_gate_al_prefix(tmp_path):
    out = tmp_path / "materials_property_al.json"
    cmd = [
        sys.executable,
        "scripts/materials_property_gate.py",
        "--fixture",
        "examples/interop/materials_parity_suite_v2.json",
        "--config",
        "examples/td_1d_morse.yaml",
        "--case-prefix",
        "eam_al_",
        "--out",
        str(out),
        "--strict",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["ok_all"] is True
    assert int(data["total"]) >= 3


def test_materials_property_gate_alloy_prefix(tmp_path):
    out = tmp_path / "materials_property_alloy.json"
    cmd = [
        sys.executable,
        "scripts/materials_property_gate.py",
        "--fixture",
        "examples/interop/materials_parity_suite_v2.json",
        "--config",
        "examples/td_1d_morse.yaml",
        "--case-prefix",
        "eam_alloy_",
        "--out",
        str(out),
        "--strict",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["ok_all"] is True
    assert int(data["total"]) >= 3
