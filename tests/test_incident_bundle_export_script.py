from __future__ import annotations

import os
import subprocess
import sys


def test_export_incident_bundle_script(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text('{"cfg":1}', encoding="utf-8")
    (run_dir / "summary.json").write_text('{"ok":false}', encoding="utf-8")
    (run_dir / "summary.md").write_text("# summary", encoding="utf-8")
    out_zip = tmp_path / "bundle.zip"

    cmd = [
        sys.executable,
        "scripts/export_incident_bundle.py",
        "--run-dir",
        str(run_dir),
        "--reason",
        "pytest",
        "--zip-out",
        str(out_zip),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out_zip.is_file()
    assert os.path.isdir(run_dir / "incident_bundle_manual")
