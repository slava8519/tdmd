from __future__ import annotations

import json
import subprocess
import sys


def test_bench_cluster_scale_simulated(tmp_path):
    out_csv = tmp_path / "cluster_scale.csv"
    out_md = tmp_path / "cluster_scale.md"
    out_json = tmp_path / "cluster_scale_summary.json"
    cmd = [
        sys.executable,
        "scripts/bench_cluster_scale.py",
        "--profile",
        "examples/cluster/cluster_profile_smoke.yaml",
        "--simulate",
        "--strict",
        "--out",
        str(out_csv),
        "--md",
        str(out_md),
        "--json",
        str(out_json),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["ok_all"] is True
    assert int(data["total"]) >= 4


def test_bench_cluster_stability_simulated(tmp_path):
    out_csv = tmp_path / "cluster_stability.csv"
    out_md = tmp_path / "cluster_stability.md"
    out_json = tmp_path / "cluster_stability_summary.json"
    cmd = [
        sys.executable,
        "scripts/bench_cluster_stability.py",
        "--profile",
        "examples/cluster/cluster_profile_smoke.yaml",
        "--simulate",
        "--strict",
        "--out",
        str(out_csv),
        "--md",
        str(out_md),
        "--json",
        str(out_json),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["ok_all"] is True
    assert int(data["total"]) >= 4


def test_bench_transport_matrix_simulated(tmp_path):
    out_csv = tmp_path / "transport_matrix.csv"
    out_md = tmp_path / "transport_matrix.md"
    out_json = tmp_path / "transport_matrix_summary.json"
    cmd = [
        sys.executable,
        "scripts/bench_mpi_transport_matrix.py",
        "--profile",
        "examples/cluster/cluster_profile_smoke.yaml",
        "--simulate",
        "--strict",
        "--out",
        str(out_csv),
        "--md",
        str(out_md),
        "--json",
        str(out_json),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["ok_all"] is True
    assert int(data["total"]) >= 6
