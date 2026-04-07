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


def test_verifylab_td_autozoning_advisor_gpu_preset_contract():
    from scripts.run_verifylab_matrix import PRESETS

    preset = dict(PRESETS["td_autozoning_advisor_gpu"])
    assert bool(preset.get("td_autozoning_advisor_mode")) is True
    assert int(preset.get("n_atoms", 0)) == 4096
    assert int(preset.get("steps", 0)) == 64
    assert int(preset.get("repeats", 0)) == 1
    assert int(preset.get("warmup", 0)) == 1
    assert int(preset.get("zone_cells_w", 0)) == 1
    assert int(preset.get("zone_cells_s", 0)) == 2
    assert bool(preset.get("require_effective_cuda")) is True
    assert str(preset.get("artifact_stem", "")) == "td_autozoning_advisor_gpu"


def test_verifylab_al_crack_100k_compare_gpu_preset_contract():
    from scripts.run_verifylab_matrix import PRESETS

    preset = dict(PRESETS["al_crack_100k_compare_gpu"])
    assert bool(preset.get("al_crack_compare_mode")) is True
    assert str(preset.get("device", "")) == "cuda"
    assert int(preset.get("timeout", 0)) == 600
    assert int(preset.get("target_atoms", 0)) == 100000
    assert float(preset.get("box", 0.0)) == 122.0
    assert int(preset.get("steps", 0)) == 100
    assert int(preset.get("requested_zones", 0)) == 1000
    assert float(preset.get("cell_size", 0.0)) == 0.5
    assert int(preset.get("telemetry_every", 0)) == 1
    assert float(preset.get("telemetry_heartbeat_sec", 0.0)) == 5.0
    assert bool(preset.get("telemetry_stdout")) is True
    assert bool(preset.get("require_effective_cuda")) is True
    assert str(preset.get("artifact_stem", "")) == "al_crack_100k_compare_gpu"


def test_verifylab_slab_wavefront_evidence_gpu_preset_contract():
    from scripts.run_verifylab_matrix import PRESETS

    preset = dict(PRESETS["slab_wavefront_evidence_gpu"])
    assert bool(preset.get("slab_wavefront_evidence_mode")) is True
    assert str(preset.get("device", "")) == "cuda"
    assert int(preset.get("timeout", 0)) == 7200
    assert int(preset.get("target_atoms", 0)) == 100000
    assert int(preset.get("exact_requested_zones", 0)) == 1000
    assert str(preset.get("crack_zones", "")) == "1,2,3,4,5,6,7,8,9,10,11,12"
    assert str(preset.get("control_zone_totals", "")) == "1,2,3,4,5,6,7,8"
    assert int(preset.get("control_breakdown_zones_total", 0)) == 8
    assert int(preset.get("exact_timeout_sec", 0)) == 900
    assert int(preset.get("crack_sweep_timeout_sec", 0)) == 600
    assert int(preset.get("requested_space_timeout_sec", 0)) == 420
    assert int(preset.get("compare_space_timeout_sec", 0)) == 420
    assert int(preset.get("control_n_atoms", 0)) == 10000
    assert int(preset.get("control_steps", 0)) == 256
    assert bool(preset.get("require_effective_cuda")) is True
    assert str(preset.get("artifact_stem", "")) == "slab_wavefront_evidence_gpu"


def test_verifylab_wavefront_reference_smoke_preset_contract():
    from scripts.run_verifylab_matrix import PRESETS

    preset = dict(PRESETS["wavefront_reference_smoke"])
    assert bool(preset.get("wavefront_reference_mode")) is True
    assert int(preset.get("timeout", 0)) == 900
    assert str(preset.get("morse_config", "")) == "examples/td_1d_morse.yaml"
    assert int(preset.get("morse_steps", 0)) == 4
    assert int(preset.get("morse_zones_total", 0)) == 8
    assert int(preset.get("eam_alloy_n_atoms", 0)) == 1000
    assert int(preset.get("eam_alloy_steps", 0)) == 2
    assert int(preset.get("eam_alloy_zones_total", 0)) == 4
    assert float(preset.get("eam_alloy_cell_size", 0.0)) == 3.5
    assert str(preset.get("eam_alloy_file", "")) == "examples/potentials/eam_alloy/AlCu.eam.alloy"
    assert bool(preset.get("allow_no_multi_zone_wave", True)) is False
    assert str(preset.get("artifact_stem", "")) == "wavefront_reference_equivalence"
