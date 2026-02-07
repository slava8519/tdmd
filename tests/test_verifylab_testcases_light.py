import json
import os
import subprocess
import sys


def test_verifylab_testcases_light_smoke():
    run_id = "pytest_testcases_light"
    cmd = [
        sys.executable,
        "scripts/run_verifylab_matrix.py",
        "examples/td_1d_lj.yaml",
        "--preset",
        "paper_testcases_light",
        "--cases-mode",
        "testcases",
        "--run-id",
        run_id,
    ]
    subprocess.check_call(cmd)
    summary_path = os.path.join("results", run_id, "summary.json")
    assert os.path.exists(summary_path), f"missing summary: {summary_path}"
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("total", 0) > 0
