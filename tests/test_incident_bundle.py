from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid

from tdmd.incident_bundle import export_incident_bundle_zip, write_incident_bundle


def test_write_incident_bundle_manifest(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text('{"x":1}', encoding="utf-8")
    (run_dir / "summary.json").write_text('{"ok":false}', encoding="utf-8")
    (run_dir / "summary.md").write_text("# summary", encoding="utf-8")

    bundle_dir = write_incident_bundle(str(run_dir), reason="unit_test", extra={"k": "v"})
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["reason"] == "unit_test"
    assert manifest["extra"]["k"] == "v"
    assert len(manifest["files"]) >= 3

    zip_path = export_incident_bundle_zip(bundle_dir, str(tmp_path / "bundle.zip"))
    assert os.path.isfile(zip_path)


def test_verifylab_strict_failure_creates_incident_bundle(tmp_path):
    run_id = f"pytest_incident_fail_{uuid.uuid4().hex[:8]}"
    outdir = tmp_path / "results"
    env_file = tmp_path / "envelope_fail.json"
    env_file.write_text(
        json.dumps(
            {
                "envelope_version": 1,
                "rows": [
                    {
                        "case": "cfg_system",
                        "zones_total": 4,
                        "use_verlet": False,
                        "verlet_k_steps": 10,
                        "chaos_mode": False,
                        "chaos_delay_prob": 0.0,
                        "max": {
                            "final_dr": 0.0,
                            "final_dv": 0.0,
                            "final_dE": 0.0,
                            "final_dT": 0.0,
                            "final_dP": 0.0,
                            "rms_dr": 0.0,
                            "rms_dv": 0.0,
                            "rms_dE": 0.0,
                            "rms_dT": 0.0,
                            "rms_dP": 0.0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        "scripts/run_verifylab_matrix.py",
        "examples/td_1d_morse.yaml",
        "--preset",
        "smoke_ci",
        "--strict",
        "--envelope-file",
        str(env_file),
        "--outdir",
        str(outdir),
        "--run-id",
        run_id,
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode != 0, proc.stdout

    summary_path = outdir / run_id / "summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert bool(summary.get("ok_all")) is False
    ib = dict(summary.get("incident_bundle", {}))
    assert ib.get("reason") == "strict_failure"
    bundle_dir = ib.get("path", "")
    assert bundle_dir and os.path.isdir(bundle_dir)
    assert os.path.isfile(os.path.join(bundle_dir, "manifest.json"))
