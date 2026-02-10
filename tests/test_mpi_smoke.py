from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _run_mpi_smoke_case(*, n: int, config: str):
    if os.environ.get("TDMD_MPI_SMOKE", "") != "1":
        pytest.skip("set TDMD_MPI_SMOKE=1 to enable MPI smoke")
    mpirun = shutil.which("mpiexec.hydra") or shutil.which("mpiexec") or shutil.which("mpirun")
    if not mpirun:
        pytest.skip("mpirun/mpiexec not available")
    try:
        import mpi4py  # noqa: F401
    except Exception:
        pytest.skip("mpi4py not available")

    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "scripts/run_mpi_smoke.py",
        "--n",
        str(int(n)),
        "--mpirun",
        str(mpirun),
        "--config",
        str(config),
    ]
    res = subprocess.run(cmd, cwd=str(root), env=os.environ.copy(), timeout=120)
    assert res.returncode == 0


def test_mpi_smoke_static_rr():
    _run_mpi_smoke_case(n=2, config="examples/td_1d_morse_static_rr.yaml")


def test_mpi_smoke_static_rr_n4():
    _run_mpi_smoke_case(n=4, config="examples/td_1d_morse_static_rr_smoke4.yaml")
