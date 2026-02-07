from __future__ import annotations
import subprocess
import sys

def test_golden_smoke_check():
    cmd = [sys.executable, "scripts/run_verifylab_matrix.py", "examples/td_1d_morse.yaml",
           "--preset", "smoke", "--golden", "check", "--run-id", "pytest"]
    subprocess.check_call(cmd)
