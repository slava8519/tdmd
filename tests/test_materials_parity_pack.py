from __future__ import annotations

import json
import subprocess
import sys


def test_materials_parity_pack_strict(tmp_path):
    out = tmp_path / "materials_parity_summary.json"
    cmd = [
        sys.executable,
        "scripts/materials_parity_pack.py",
        "--fixture",
        "examples/interop/materials_parity_suite_v1.json",
        "--config",
        "examples/td_1d_morse.yaml",
        "--out",
        str(out),
        "--strict",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["ok_all"] is True
    assert int(data["total"]) >= 2


def test_materials_parity_pack_v2_strict(tmp_path):
    out = tmp_path / "materials_parity_v2_summary.json"
    cmd = [
        sys.executable,
        "scripts/materials_parity_pack.py",
        "--fixture",
        "examples/interop/materials_parity_suite_v2.json",
        "--config",
        "examples/td_1d_morse.yaml",
        "--out",
        str(out),
        "--strict",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["ok_all"] is True
    assert int(data["total"]) >= 6
    by_prop = dict(data.get("by_property", {}))
    assert "eos" in by_prop
    assert "thermo" in by_prop
    assert "transport" in by_prop
