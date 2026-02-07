from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def _find_mpirun(user_arg: str | None) -> str | None:
    if user_arg:
        return user_arg
    env_val = os.environ.get("MPIRUN", "").strip()
    if env_val:
        return env_val
    return (
        shutil.which("mpiexec.hydra")
        or shutil.which("mpiexec")
        or shutil.which("mpirun")
    )


def main() -> int:
    p = argparse.ArgumentParser(description="MPI smoke test runner")
    p.add_argument("--n", type=int, default=2, help="MPI ranks (default: 2)")
    p.add_argument("--config", default="examples/td_1d_morse_static_rr.yaml")
    p.add_argument("--mpirun", default="", help="Path to mpirun/mpiexec")
    p.add_argument("--timeout", type=int, default=60)
    args = p.parse_args()

    mpirun = _find_mpirun(args.mpirun or None)
    if not mpirun:
        print("[mpi-smoke] mpirun/mpiexec not found", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parents[1]
    cfg = Path(args.config)
    if not cfg.is_absolute():
        cfg = root / cfg
    if not cfg.exists():
        print(f"[mpi-smoke] config not found: {cfg}", file=sys.stderr)
        return 2

    cmd = [mpirun, "-n", str(int(args.n)), sys.executable, "-m", "tdmd.main", "run", str(cfg)]
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    print("[mpi-smoke] " + " ".join(cmd), flush=True)
    try:
        res = subprocess.run(cmd, cwd=str(root), env=env, timeout=int(args.timeout))
    except subprocess.TimeoutExpired:
        print("[mpi-smoke] timed out", file=sys.stderr)
        return 3
    return int(res.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
