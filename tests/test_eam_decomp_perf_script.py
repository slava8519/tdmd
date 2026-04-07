from __future__ import annotations

import json
import subprocess
import sys


def test_eam_decomp_perf_script_writes_artifacts(tmp_path):
    out_csv = tmp_path / "eam_decomp_perf.csv"
    out_md = tmp_path / "eam_decomp_perf.md"
    out_json = tmp_path / "eam_decomp_perf.summary.json"
    cmd = [
        sys.executable,
        "scripts/bench_eam_decomp_perf.py",
        "--out",
        str(out_csv),
        "--md",
        str(out_md),
        "--json",
        str(out_json),
        "--n-atoms",
        "256",
        "--steps",
        "1",
        "--repeats",
        "1",
        "--warmup",
        "0",
        "--zones-total",
        "2",
        "--zones-nx",
        "2",
        "--zones-ny",
        "1",
        "--zones-nz",
        "1",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out_csv.is_file()
    assert out_md.is_file()
    assert out_json.is_file()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert bool(data.get("ok_all")) is True
    assert int(data.get("total", 0)) == 4
    assert sorted(dict(data.get("by_case", {})).keys()) == [
        "space_cpu",
        "space_gpu",
        "time_cpu",
        "time_gpu",
    ]
    time_wave = dict(dict(data.get("by_case", {})).get("time_gpu", {})).get(
        "wave_batch_diagnostics", {}
    )
    assert str(dict(time_wave).get("version", "")) == "pr_sw05_v1"
    assert "launches_saved_per_step" in dict(time_wave)
    assert "neighbor_reuse_ratio_weighted" in dict(time_wave)

    report = out_md.read_text(encoding="utf-8")
    assert "EAM Alloy Decomposition Benchmark" in report
    assert "| metric | space_cpu | space_gpu | time_cpu | time_gpu |" in report
    assert "gpu_speedup_vs_cpu" in report
    assert "td_speedup_vs_space" in report
    assert "EAM Alloy Decomposition Benchmark" in proc.stdout


def test_eam_decomp_perf_script_supports_gpu_only_case_selection(tmp_path):
    out_csv = tmp_path / "eam_decomp_perf_gpu_only.csv"
    out_md = tmp_path / "eam_decomp_perf_gpu_only.md"
    out_json = tmp_path / "eam_decomp_perf_gpu_only.summary.json"
    cmd = [
        sys.executable,
        "scripts/bench_eam_decomp_perf.py",
        "--out",
        str(out_csv),
        "--md",
        str(out_md),
        "--json",
        str(out_json),
        "--n-atoms",
        "256",
        "--steps",
        "1",
        "--repeats",
        "1",
        "--warmup",
        "0",
        "--zones-total",
        "2",
        "--zones-nx",
        "2",
        "--zones-ny",
        "1",
        "--zones-nz",
        "1",
        "--cases",
        "space_gpu,time_gpu",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out_json.is_file()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert bool(data.get("ok_all")) is True
    assert int(data.get("total", 0)) == 2
    assert list(data.get("selected_cases", [])) == ["space_gpu", "time_gpu"]
    assert sorted(dict(data.get("by_case", {})).keys()) == ["space_gpu", "time_gpu"]
    assert (
        str(
            dict(dict(data.get("by_case", {})).get("time_gpu", {}))
            .get("wave_batch_diagnostics", {})
            .get("version", "")
        )
        == "pr_sw05_v1"
    )

    report = out_md.read_text(encoding="utf-8")
    assert "| metric | space_gpu | time_gpu |" in report
    assert "td_speedup_vs_space" in report
