from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _detect_mpirun_for_test() -> str:
    env_val = os.environ.get("MPIRUN", "").strip()
    if env_val:
        return env_val
    root = Path(__file__).resolve().parents[1]
    for cand in (
        root / ".venv" / "bin" / "mpiexec.hydra",
        root / ".venv" / "bin" / "mpiexec",
        root / ".venv" / "bin" / "mpirun",
    ):
        if cand.is_file() and os.access(str(cand), os.X_OK):
            return str(cand)
    return shutil.which("mpiexec.hydra") or shutil.which("mpiexec") or shutil.which("mpirun") or ""


def test_verifylab_mpi_overlap_smoke_strict():
    if os.environ.get("TDMD_MPI_SMOKE", "") != "1":
        pytest.skip("set TDMD_MPI_SMOKE=1 to enable MPI overlap smoke")
    if not _detect_mpirun_for_test():
        pytest.skip("mpirun/mpiexec not available")
    run_id = "pytest_mpi_overlap_smoke"
    cmd = [
        sys.executable,
        "scripts/run_verifylab_matrix.py",
        "examples/td_1d_morse.yaml",
        "--preset",
        "mpi_overlap_smoke",
        "--strict",
        "--run-id",
        run_id,
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    summary_path = Path("results") / run_id / "summary.json"
    assert summary_path.is_file()
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    assert bool(summary.get("ok_all")) is True
    runs = list(summary.get("mpi_overlap_runs", []))
    assert len(runs) == 2
    ranks = sorted(int(r.get("ranks", 0)) for r in runs)
    assert ranks == [2, 4]
