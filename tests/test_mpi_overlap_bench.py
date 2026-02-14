from __future__ import annotations

import csv
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


def test_bench_mpi_overlap_simulated_outputs_async_columns(tmp_path):
    out_csv = tmp_path / "mpi_overlap.csv"
    out_md = tmp_path / "mpi_overlap.md"
    cmd = [
        sys.executable,
        "scripts/bench_mpi_overlap.py",
        "--config",
        "examples/td_1d_morse_static_rr_smoke4.yaml",
        "--overlap-list",
        "0,1",
        "--simulate",
        "--require-async-evidence",
        "--out",
        str(out_csv),
        "--md",
        str(out_md),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out_csv.is_file()
    with open(out_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None
        assert "async_send_msgs_max" in reader.fieldnames
        assert "async_send_bytes_max" in reader.fieldnames
        assert "async_evidence_ok" in reader.fieldnames
        rows = list(reader)
        overlap1 = [r for r in rows if r.get("overlap") == "1"]
        assert overlap1
        assert all(int(r.get("async_evidence_ok", "0")) == 1 for r in overlap1)
