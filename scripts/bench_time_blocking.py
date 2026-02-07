from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Iterable

import yaml


def _find_mpirun(user_arg: str | None) -> str | None:
    if user_arg:
        return user_arg
    env_val = os.environ.get("MPIRUN", "").strip()
    if env_val:
        return env_val
    return shutil.which("mpirun") or shutil.which("mpiexec")


def _parse_k_list(text: str) -> list[int]:
    vals: list[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        k = int(token)
        if k < 1:
            raise ValueError("time-blocking K values must be >= 1")
        vals.append(k)
    if not vals:
        raise ValueError("k-list must contain at least one positive integer")
    return vals


def _write_config_with_k(base_cfg: dict, k: int, steps: int | None) -> str:
    cfg = dict(base_cfg)
    td = dict(cfg.get("td", {}))
    td["time_block_k"] = int(k)
    td["batch_size"] = int(k)
    cfg["td"] = td
    if steps is not None:
        run = dict(cfg.get("run", {}))
        run["n_steps"] = int(steps)
        run["thermo_every"] = max(1, min(int(run.get("thermo_every", int(steps))), int(steps)))
        cfg["run"] = run

    fd, tmp_path = tempfile.mkstemp(prefix=f"tdmd_k{k}_", suffix=".yaml")
    os.close(fd)
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return tmp_path


def _iter_bench_rows(
    *,
    mpirun: str,
    n_ranks: int,
    python_exe: str,
    root: Path,
    base_cfg: dict,
    k_list: Iterable[int],
    steps: int | None,
    timeout: int,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for k in k_list:
        cfg_tmp = _write_config_with_k(base_cfg, int(k), steps)
        cmd = [mpirun, "-n", str(int(n_ranks)), python_exe, "-m", "tdmd.main", "run", cfg_tmp]
        t0 = time.perf_counter()
        try:
            res = subprocess.run(
                cmd,
                cwd=str(root),
                timeout=int(timeout),
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            elapsed = float(time.perf_counter() - t0)
            rows.append(
                {
                    "k": int(k),
                    "returncode": int(res.returncode),
                    "elapsed_sec": elapsed,
                    "stdout_bytes": int(len(res.stdout.encode("utf-8"))),
                    "stderr_bytes": int(len(res.stderr.encode("utf-8"))),
                }
            )
        finally:
            try:
                os.unlink(cfg_tmp)
            except OSError:
                pass
    return rows


def _write_csv(path: str, rows: list[dict[str, float | int]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k", "returncode", "elapsed_sec", "speedup_vs_k1", "stdout_bytes", "stderr_bytes"])
        base = None
        for row in rows:
            if int(row["returncode"]) == 0 and int(row["k"]) == 1:
                base = float(row["elapsed_sec"])
                break
        for row in rows:
            elapsed = float(row["elapsed_sec"])
            speedup = (base / elapsed) if (base is not None and elapsed > 0.0 and int(row["returncode"]) == 0) else 0.0
            w.writerow([
                int(row["k"]),
                int(row["returncode"]),
                f"{elapsed:.6f}",
                f"{speedup:.6f}",
                int(row["stdout_bytes"]),
                int(row["stderr_bytes"]),
            ])


def main() -> int:
    p = argparse.ArgumentParser(description="Measure TD-MPI time-blocking K impact (batch_size/time_block_k)")
    p.add_argument("--config", default="examples/td_1d_morse_static_rr.yaml")
    p.add_argument("--n", type=int, default=2, help="MPI ranks")
    p.add_argument("--k-list", default="1,2,4", help="Comma-separated K values (>=1)")
    p.add_argument("--steps", type=int, default=40, help="Override run.n_steps (set <=0 to keep config value)")
    p.add_argument("--mpirun", default="", help="Path to mpirun/mpiexec")
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--out", default="results/time_blocking.csv")
    p.add_argument("--dry-run", action="store_true", help="Only validate inputs and print planned runs")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    if not cfg_path.exists():
        print(f"[bench-time-blocking] config not found: {cfg_path}", file=sys.stderr)
        return 2

    try:
        k_list = _parse_k_list(args.k_list)
    except Exception as exc:
        print(f"[bench-time-blocking] invalid --k-list: {exc}", file=sys.stderr)
        return 2

    with open(cfg_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    steps = int(args.steps)
    if steps <= 0:
        steps = None

    mpirun = _find_mpirun(args.mpirun or None)
    if args.dry_run:
        print(f"[bench-time-blocking] dry-run config={cfg_path}")
        print(f"[bench-time-blocking] ranks={int(args.n)} k_list={k_list} steps_override={steps}")
        print(f"[bench-time-blocking] mpirun={'<auto>' if not args.mpirun else args.mpirun}")
        return 0

    if not mpirun:
        print("[bench-time-blocking] mpirun/mpiexec not found", file=sys.stderr)
        return 2

    rows = _iter_bench_rows(
        mpirun=str(mpirun),
        n_ranks=int(args.n),
        python_exe=sys.executable,
        root=root,
        base_cfg=base_cfg,
        k_list=k_list,
        steps=steps,
        timeout=int(args.timeout),
    )
    _write_csv(str(args.out), rows)

    print(f"[bench-time-blocking] wrote {args.out}")
    for row in rows:
        print(
            f"  k={int(row['k'])} rc={int(row['returncode'])} elapsed={float(row['elapsed_sec']):.3f}s "
            f"stdout={int(row['stdout_bytes'])}B stderr={int(row['stderr_bytes'])}B"
        )

    any_ok = any(int(r["returncode"]) == 0 for r in rows)
    all_ok = all(int(r["returncode"]) == 0 for r in rows)
    if not any_ok:
        return 3
    return 0 if all_ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
