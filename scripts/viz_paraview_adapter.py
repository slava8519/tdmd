#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare/run ParaView adapter artifacts")
    ap.add_argument("--traj", required=True, help="Input trajectory path")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--run", action="store_true", help="Execute generated pvpython script if available")
    ap.add_argument("--pvpython", default="", help="Path to pvpython binary (optional)")
    args = ap.parse_args()

    outdir = str(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    script_path = os.path.join(outdir, "paraview_pipeline.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(
            "from paraview.simple import *\n\n"
            "# Placeholder pipeline. Adjust reader/filter stack to your local ParaView plugins.\n"
            f"traj_path = r'''{os.path.abspath(args.traj)}'''\n"
            "print('paraview_input', traj_path)\n"
            "print('paraview_note', 'Configure LAMMPS dump reader/plugin in this script')\n"
        )

    pvpython = str(args.pvpython).strip() or shutil.which("pvpython") or ""
    status = "prepared"
    rc = 0
    run_cmd = []
    stdout = ""
    stderr = ""
    if args.run:
        if not pvpython:
            status = "skipped_missing_pvpython"
        else:
            run_cmd = [pvpython, script_path]
            proc = subprocess.run(run_cmd, capture_output=True, text=True)
            rc = int(proc.returncode)
            stdout = str(proc.stdout or "")
            stderr = str(proc.stderr or "")
            status = "ran" if rc == 0 else "run_failed"

    payload = {
        "ok": bool(status in ("prepared", "ran")),
        "status": status,
        "traj": os.path.abspath(args.traj),
        "script": os.path.abspath(script_path),
        "run": bool(args.run),
        "command": run_cmd,
        "returncode": int(rc),
        "stdout": stdout,
        "stderr": stderr,
    }
    out_json = os.path.join(outdir, "paraview_adapter.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

