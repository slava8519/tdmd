from __future__ import annotations

import subprocess
import sys


def test_profile_gpu_backend_script(tmp_path):
    out_csv = tmp_path / "gpu_profile.csv"
    out_md = tmp_path / "gpu_profile.md"
    cmd = [
        sys.executable,
        "scripts/profile_gpu_backend.py",
        "--config",
        "examples/td_1d_morse.yaml",
        "--out-csv",
        str(out_csv),
        "--out-md",
        str(out_md),
        "--timeout",
        "120",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out_csv.is_file()
    assert out_md.is_file()
    text = out_md.read_text(encoding="utf-8")
    assert "GPU Backend Profile" in text
