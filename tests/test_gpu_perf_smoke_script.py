from __future__ import annotations

import json
import subprocess
import sys


def test_gpu_perf_smoke_script_writes_summary(tmp_path):
    out_json = tmp_path / "gpu_perf_smoke.summary.json"
    cmd = [
        sys.executable,
        "scripts/bench_gpu_perf_smoke.py",
        "--out-json",
        str(out_json),
        "--repeats",
        "3",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out_json.is_file()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "ok_all" in data
    assert "total" in data
