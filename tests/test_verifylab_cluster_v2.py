from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_preset(preset: str, run_id: str) -> dict:
    cmd = [
        sys.executable,
        "scripts/run_verifylab_matrix.py",
        "examples/td_1d_morse.yaml",
        "--preset",
        preset,
        "--strict",
        "--run-id",
        run_id,
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    path = Path("results") / run_id / "summary.json"
    assert path.is_file()
    return json.loads(path.read_text(encoding="utf-8"))


def test_verifylab_cluster_scale_smoke_strict():
    data = _run_preset("cluster_scale_smoke", "pytest_cluster_scale_smoke")
    assert bool(data.get("ok_all")) is True
    assert int(data.get("total", 0)) >= 4


def test_verifylab_cluster_stability_smoke_strict():
    data = _run_preset("cluster_stability_smoke", "pytest_cluster_stability_smoke")
    assert bool(data.get("ok_all")) is True
    assert int(data.get("total", 0)) >= 4


def test_verifylab_transport_matrix_smoke_strict():
    data = _run_preset("mpi_transport_matrix_smoke", "pytest_transport_matrix_smoke")
    assert bool(data.get("ok_all")) is True
    assert int(data.get("total", 0)) >= 6
