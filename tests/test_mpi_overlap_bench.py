from __future__ import annotations

import subprocess
import sys


def test_bench_mpi_overlap_dry_run():
    cmd = [
        sys.executable,
        "scripts/bench_mpi_overlap.py",
        "--config",
        "examples/td_1d_morse_static_rr_smoke4.yaml",
        "--overlap-list",
        "0,1",
        "--cuda-aware",
        "--dry-run",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "overlap_modes=[0, 1]" in proc.stdout
    assert "cuda_aware=True" in proc.stdout


def test_bench_mpi_overlap_dry_run_with_profile():
    cmd = [
        sys.executable,
        "scripts/bench_mpi_overlap.py",
        "--profile",
        "examples/cluster/cluster_profile_smoke.yaml",
        "--dry-run",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "overlap_modes=[0, 1]" in proc.stdout
