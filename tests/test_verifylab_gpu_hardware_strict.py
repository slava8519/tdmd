from __future__ import annotations

import json
import os
import subprocess
import sys

from tdmd.backend import resolve_backend


def test_verifylab_gpu_smoke_hw_backend_gate():
    run_id = "pytest_gpu_smoke_hw"
    cmd = [
        sys.executable,
        "scripts/run_verifylab_matrix.py",
        "examples/td_1d_morse.yaml",
        "--preset",
        "gpu_smoke_hw",
        "--strict",
        "--run-id",
        run_id,
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    summary_path = os.path.join("results", run_id, "summary.json")
    assert os.path.exists(summary_path), proc.stderr or proc.stdout
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    backend = dict(summary.get("backend", {}))
    assert backend.get("requested_device") == "cuda"
    assert "effective_device" in backend
    assert summary.get("require_effective_cuda") is True

    have_cuda = resolve_backend("cuda").device == "cuda"
    if have_cuda:
        assert proc.returncode == 0, proc.stderr or proc.stdout
        assert bool(summary.get("backend_ok")) is True
        assert bool(backend.get("fallback_from_cuda")) is False
    else:
        assert proc.returncode != 0
        assert bool(summary.get("backend_ok")) is False
        assert bool(backend.get("fallback_from_cuda")) is True
