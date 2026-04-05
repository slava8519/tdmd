from __future__ import annotations

import json
import subprocess
import sys


def test_ml_reference_parity_pack_strict(tmp_path):
    out = tmp_path / "ml_reference_parity_summary.json"
    cmd = [
        sys.executable,
        "scripts/ml_reference_parity_pack.py",
        "--fixture",
        "examples/interop/ml_reference_suite_v1.json",
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
    assert int(data["total"]) == 2
    assert data["family"] == "ml/reference:quadratic_density"


def test_verifylab_ml_reference_presets_contract():
    from scripts.run_verifylab_matrix import PRESETS

    smoke = dict(PRESETS["ml_reference_smoke"])
    assert bool(smoke.get("task_mode")) is True
    assert bool(smoke.get("sync_mode")) is True
    assert str(smoke.get("task_path", "")) == "examples/interop/task_ml_reference.yaml"
    assert list(smoke.get("zones_total_list", [])) == [2]

    parity = dict(PRESETS["ml_reference_parity_smoke"])
    assert bool(parity.get("ml_reference_parity_mode")) is True
    assert (
        str(parity.get("ml_reference_fixture", "")) == "examples/interop/ml_reference_suite_v1.json"
    )
