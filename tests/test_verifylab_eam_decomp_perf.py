from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_verifylab_eam_decomp_perf_smoke_strict():
    run_id = "pytest_eam_decomp_perf_smoke"
    cmd = [
        sys.executable,
        "scripts/run_verifylab_matrix.py",
        "examples/td_1d_morse.yaml",
        "--preset",
        "eam_decomp_perf_smoke",
        "--strict",
        "--run-id",
        run_id,
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    out_dir = Path("results") / run_id
    summary_path = out_dir / "summary.json"
    summary_md_path = out_dir / "summary.md"
    bench_md_path = out_dir / "eam_decomp_perf.md"
    assert summary_path.is_file()
    assert summary_md_path.is_file()
    assert bench_md_path.is_file()

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert bool(data.get("ok_all")) is True
    assert int(data.get("total", 0)) == 4
    assert "report_markdown" in data
    assert "EAM Alloy Decomposition Benchmark" in str(data.get("report_markdown", ""))

    summary_md = summary_md_path.read_text(encoding="utf-8")
    assert "EAM Alloy Decomposition Benchmark" in summary_md
    assert "| metric | space_cpu | space_gpu | time_cpu | time_gpu |" in summary_md

    assert "EAM Alloy Decomposition Benchmark" in proc.stdout
    assert "| metric | space_cpu | space_gpu | time_cpu | time_gpu |" in proc.stdout


def test_verifylab_eam_decomp_perf_gpu_heavy_preset_contract():
    from scripts.run_verifylab_matrix import PRESETS

    preset = dict(PRESETS["eam_decomp_perf_gpu_heavy"])
    assert bool(preset.get("eam_decomp_perf_mode")) is True
    assert int(preset.get("n_atoms", 0)) == 10000
    assert int(preset.get("steps", 0)) == 768
    assert bool(preset.get("require_effective_cuda")) is True
    assert int(preset.get("zones_total", 0)) == 6
    assert int(preset.get("zones_nx", 0)) == 3
    assert int(preset.get("zones_ny", 0)) == 2
    assert int(preset.get("zones_nz", 0)) == 1
    assert list(preset.get("cases", [])) == ["space_gpu", "time_gpu"]
    assert str(preset.get("artifact_stem", "")) == "eam_decomp_perf_gpu_heavy"


def test_verifylab_eam_decomp_zone_sweep_gpu_preset_contract():
    from scripts.run_verifylab_matrix import PRESETS

    preset = dict(PRESETS["eam_decomp_zone_sweep_gpu"])
    assert bool(preset.get("eam_decomp_zone_sweep_mode")) is True
    assert int(preset.get("n_atoms", 0)) == 10000
    assert int(preset.get("steps", 0)) == 256
    assert bool(preset.get("require_effective_cuda")) is True
    assert str(preset.get("layouts", "")) == "2:2x1x1,4:2x2x1,6:3x2x1"
    assert str(preset.get("artifact_stem", "")) == "eam_decomp_zone_sweep_gpu"


def test_verifylab_eam_td_breakdown_gpu_preset_contract():
    from scripts.run_verifylab_matrix import PRESETS

    preset = dict(PRESETS["eam_td_breakdown_gpu"])
    assert bool(preset.get("eam_td_breakdown_mode")) is True
    assert int(preset.get("n_atoms", 0)) == 10000
    assert int(preset.get("steps", 0)) == 768
    assert bool(preset.get("require_effective_cuda")) is True
    assert int(preset.get("zones_total", 0)) == 2
    assert int(preset.get("zones_nx", 0)) == 2
    assert int(preset.get("zones_ny", 0)) == 1
    assert int(preset.get("zones_nz", 0)) == 1
    assert str(preset.get("artifact_stem", "")) == "eam_td_breakdown_gpu"
