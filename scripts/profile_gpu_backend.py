from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _run(cmd: list[str], cwd: str, timeout: int) -> tuple[int, float, str, str]:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    dt = float(time.perf_counter() - t0)
    return int(proc.returncode), dt, proc.stdout, proc.stderr


def _read_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    p = argparse.ArgumentParser(description="Profile CPU vs GPU-track verify presets")
    p.add_argument("--config", default="examples/td_1d_morse.yaml")
    p.add_argument("--out-csv", default="results/gpu_profile.csv")
    p.add_argument("--out-md", default="results/gpu_profile.md")
    p.add_argument("--timeout", type=int, default=300)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = str(args.config)
    rows: list[dict[str, object]] = []

    runs = [
        ("smoke_ci", "profile_cpu_smoke"),
        ("gpu_smoke", "profile_gpu_smoke"),
        ("interop_smoke", "profile_cpu_interop"),
        ("gpu_interop_smoke", "profile_gpu_interop"),
        ("interop_metal_smoke", "profile_cpu_metal"),
        ("gpu_metal_smoke", "profile_gpu_metal"),
    ]

    for preset, run_id in runs:
        cmd = [
            sys.executable,
            "scripts/run_verifylab_matrix.py",
            cfg,
            "--preset",
            preset,
            "--strict",
            "--run-id",
            run_id,
        ]
        rc, elapsed, out, err = _run(cmd, cwd=str(root), timeout=int(args.timeout))
        summary = _read_summary(root / "results" / run_id / "summary.json")
        worst = summary.get("worst", {}) if isinstance(summary, dict) else {}
        rows.append(
            {
                "preset": preset,
                "run_id": run_id,
                "returncode": rc,
                "elapsed_sec": elapsed,
                "ok_all": (
                    bool(summary.get("ok_all", False)) if isinstance(summary, dict) else False
                ),
                "max_dr": float(worst.get("max_dr", 0.0) or 0.0),
                "max_dv": float(worst.get("max_dv", 0.0) or 0.0),
                "max_dE": float(worst.get("max_dE", 0.0) or 0.0),
                "max_dT": float(worst.get("max_dT", 0.0) or 0.0),
                "max_dP": float(worst.get("max_dP", 0.0) or 0.0),
                "stdout_bytes": len(out.encode("utf-8")),
                "stderr_bytes": len(err.encode("utf-8")),
            }
        )

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "preset",
                "run_id",
                "returncode",
                "elapsed_sec",
                "ok_all",
                "max_dr",
                "max_dv",
                "max_dE",
                "max_dT",
                "max_dP",
                "stdout_bytes",
                "stderr_bytes",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["preset"],
                    r["run_id"],
                    r["returncode"],
                    f"{float(r['elapsed_sec']):.6f}",
                    int(bool(r["ok_all"])),
                    r["max_dr"],
                    r["max_dv"],
                    r["max_dE"],
                    r["max_dT"],
                    r["max_dP"],
                    r["stdout_bytes"],
                    r["stderr_bytes"],
                ]
            )

    base = {r["preset"]: float(r["elapsed_sec"]) for r in rows}
    ratio_smoke = (
        (base.get("smoke_ci", 0.0) / max(base.get("gpu_smoke", 1e-12), 1e-12))
        if base.get("smoke_ci")
        else 0.0
    )
    ratio_interop = (
        (base.get("interop_smoke", 0.0) / max(base.get("gpu_interop_smoke", 1e-12), 1e-12))
        if base.get("interop_smoke")
        else 0.0
    )
    ratio_metal = (
        (base.get("interop_metal_smoke", 0.0) / max(base.get("gpu_metal_smoke", 1e-12), 1e-12))
        if base.get("interop_metal_smoke")
        else 0.0
    )

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("# GPU Backend Profile\n\n")
        f.write(f"- config: `{cfg}`\n")
        f.write(f"- smoke cpu/gpu ratio: `{ratio_smoke:.6f}`\n")
        f.write(f"- interop cpu/gpu ratio: `{ratio_interop:.6f}`\n")
        f.write(f"- metal cpu/gpu ratio: `{ratio_metal:.6f}`\n")
        f.write("\n## Runs\n\n")
        for r in rows:
            f.write(
                f"- `{r['preset']}` rc={r['returncode']} ok={r['ok_all']} elapsed={float(r['elapsed_sec']):.3f}s "
                f"max_dr={r['max_dr']:.3e} max_dv={r['max_dv']:.3e} max_dE={r['max_dE']:.3e}\n"
            )

    print(f"[gpu-profile] wrote {args.out_csv}")
    print(f"[gpu-profile] wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
