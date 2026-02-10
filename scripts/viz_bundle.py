#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True)
    return int(proc.returncode), str(proc.stdout or ""), str(proc.stderr or "")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run universal visualization bundle on one trajectory")
    ap.add_argument("--traj", required=True, help="Input trajectory")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument(
        "--run-adapters",
        action="store_true",
        help="Try running external adapters if binaries are available",
    )
    args = ap.parse_args()

    outdir = str(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    analyze_csv = os.path.join(outdir, "analysis.csv")
    analyze_json = os.path.join(outdir, "analysis.json")
    commands = []
    results = []

    cmd_an = [
        sys.executable,
        "scripts/viz_analyze.py",
        "--traj",
        str(args.traj),
        "--plugin",
        "mobility",
        "--plugin",
        "species_mixing",
        "--out-csv",
        analyze_csv,
        "--out-json",
        analyze_json,
    ]
    commands.append(cmd_an)
    rc, so, se = _run(cmd_an)
    results.append({"name": "analyze", "rc": rc, "stdout": so, "stderr": se})

    for nm, script in (
        ("ovito", "scripts/viz_ovito_adapter.py"),
        ("vmd", "scripts/viz_vmd_adapter.py"),
        ("paraview", "scripts/viz_paraview_adapter.py"),
    ):
        sub_out = os.path.join(outdir, nm)
        cmd = [sys.executable, script, "--traj", str(args.traj), "--outdir", sub_out]
        if args.run_adapters:
            cmd.append("--run")
        commands.append(cmd)
        rc, so, se = _run(cmd)
        results.append({"name": nm, "rc": rc, "stdout": so, "stderr": se})

    ok = bool(all(int(r["rc"]) == 0 for r in results))
    payload = {
        "ok": ok,
        "traj": os.path.abspath(args.traj),
        "outdir": os.path.abspath(outdir),
        "commands": commands,
        "results": [{k: v for k, v in r.items() if k in ("name", "rc")} for r in results],
    }
    out_json = os.path.join(outdir, "viz_bundle.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
