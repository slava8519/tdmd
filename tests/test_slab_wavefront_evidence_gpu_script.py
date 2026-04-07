from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "scripts").resolve()))

import bench_slab_wavefront_evidence_gpu as slab_wavefront_evidence_gpu


def test_slab_wavefront_evidence_gpu_script_writes_artifacts(tmp_path):
    out_csv = tmp_path / "slab_wavefront_evidence_gpu.csv"
    out_md = tmp_path / "slab_wavefront_evidence_gpu.md"
    out_json = tmp_path / "slab_wavefront_evidence_gpu.summary.json"
    cmd = [
        sys.executable,
        "scripts/bench_slab_wavefront_evidence_gpu.py",
        "--out",
        str(out_csv),
        "--md",
        str(out_md),
        "--json",
        str(out_json),
        "--target-atoms",
        "256",
        "--box",
        "40.0",
        "--steps",
        "1",
        "--device",
        "cpu",
        "--exact-requested-zones",
        "64",
        "--crack-zones",
        "1,2",
        "--control-zone-totals",
        "1,2",
        "--control-breakdown-zones-total",
        "2",
        "--cell-size",
        "2.0",
        "--control-n-atoms",
        "256",
        "--control-steps",
        "1",
        "--control-repeats",
        "1",
        "--control-warmup",
        "0",
        "--exact-timeout-sec",
        "60",
        "--crack-sweep-timeout-sec",
        "30",
        "--requested-space-timeout-sec",
        "20",
        "--compare-space-timeout-sec",
        "20",
        "--compare-time-timeout-sec",
        "20",
        "--telemetry-every",
        "1",
        "--telemetry-heartbeat-sec",
        "0.1",
        "--strict",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out_csv.is_file()
    assert out_md.is_file()
    assert out_json.is_file()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert bool(data.get("ok_all")) is True
    assert int(data.get("total", 0)) == 4
    assert "exact_compare" in data
    assert "crack_sweep" in data
    assert "control_sweep" in data
    assert "control_breakdown" in data
    assert len(list(dict(data.get("crack_sweep", {})).get("rows", []))) == 2
    assert len(list(dict(data.get("control_sweep", {})).get("rows", []))) == 2
    assert list(dict(data.get("control_sweep", {})).get("skipped_invalid_space_zone_totals", [])) == []
    assert (
        int(
            dict(dict(data.get("control_breakdown", {})).get("selected_space_layout", {})).get(
                "nx", 0
            )
        )
        >= 1
    )
    assert "control_orchestration_vs_force" in dict(data.get("evidence_views", {}))
    assert "control_wave_batch_cost_model" in dict(data.get("evidence_views", {}))
    assert (
        "time_gpu_wave_batch_launches_saved_per_step"
        in dict(data.get("control_breakdown", {}))
    )

    report = out_md.read_text(encoding="utf-8")
    assert "Slab Wavefront Evidence Pack" in report
    assert "Exact Crack Benchmark" in report
    assert "Crack Sweep `z` Trend" in report
    assert "EAM Control Sweep" in report
    assert "Control Breakdown" in report
    assert "time_gpu_wave_batch_launches_saved_per_step" in report
    assert "Slab Wavefront Evidence Pack" in proc.stdout


def test_control_sweep_skips_invalid_space_zone_totals(monkeypatch, tmp_path):
    out_dir = tmp_path / "control_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "eam_decomp_zone_sweep_gpu.summary.json"
    out_json.write_text(
        json.dumps(
            {
                "ok_all": True,
                "rows": [
                    {"zones_total": 10, "space_gpu_median_sec": 1.0, "time_gpu_median_sec": 0.8},
                    {"zones_total": 12, "space_gpu_median_sec": 1.2, "time_gpu_median_sec": 0.7},
                ],
                "artifacts": {"json": str(out_json)},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        slab_wavefront_evidence_gpu,
        "_run_subprocess",
        lambda **kwargs: (0, "", "", False),
    )

    summary = slab_wavefront_evidence_gpu._run_control_sweep(
        out_dir=out_dir,
        n_atoms=10000,
        steps=4,
        repeats=1,
        warmup=0,
        seed=42,
        cutoff=6.5,
        dt=0.001,
        cell_size=2.5,
        zone_cells_w=1,
        zone_cells_s=2,
        zone_totals=[10, 11, 12],
        eam_file="examples/potentials/eam_alloy/AlCu.eam.alloy",
        require_effective_cuda=False,
    )

    assert bool(summary.get("ok_all")) is True
    assert list(summary.get("valid_zone_totals", [])) == [10, 12]
    skipped = list(summary.get("skipped_invalid_space_zone_totals", []))
    assert [int(dict(item).get("zones_total", 0)) for item in skipped] == [11]


def test_exact_compare_evidence_accepts_requested_invalid_common_fallback(monkeypatch, tmp_path):
    out_dir = tmp_path / "exact_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "al_crack_100k_compare_gpu.summary.json"
    out_json.write_text(
        json.dumps(
            {
                "ok_all": False,
                "requested_zones_total": 1000,
                "strict_valid_common_zones_total": 18,
                "requested_td_preflight": {
                    "preflight_ok": False,
                },
                "requested_td_wavefront": {
                    "contract_version": "pr_sw01_v1",
                },
                "strict_valid_common_td_wavefront": {
                    "contract_version": "pr_sw01_v1",
                },
                "by_case": {
                    "space_gpu_z1000": {
                        "case": "space_gpu_z1000",
                        "ok": False,
                        "effective_device": "cuda",
                    },
                    "space_gpu_z18": {
                        "case": "space_gpu_z18",
                        "ok": True,
                        "effective_device": "cuda",
                    },
                    "time_gpu_z18": {
                        "case": "time_gpu_z18",
                        "ok": True,
                        "effective_device": "cuda",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        slab_wavefront_evidence_gpu,
        "_run_subprocess",
        lambda **kwargs: (0, "", "", False),
    )

    summary = slab_wavefront_evidence_gpu._run_exact_crack_compare(
        task_path=tmp_path / "task.yaml",
        out_dir=out_dir,
        device="cuda",
        requested_zones=1000,
        cell_size=0.5,
        timeout_sec=600,
        requested_space_timeout_sec=90,
        compare_space_timeout_sec=420,
        compare_time_timeout_sec=180,
        telemetry_every=1,
        telemetry_heartbeat_sec=5.0,
        telemetry_stdout=False,
        require_effective_cuda=True,
    )

    assert bool(summary.get("benchmark_ok_all")) is False
    assert bool(summary.get("requested_space_case_ok")) is False
    assert bool(summary.get("strict_valid_common_space_case_ok")) is True
    assert bool(summary.get("strict_valid_common_time_case_ok")) is True
    assert bool(summary.get("requested_td_case_required")) is False
    assert bool(summary.get("ok_all")) is True
