#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare/run VMD adapter artifacts")
    ap.add_argument("--traj", required=True, help="Input trajectory path")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument(
        "--run", action="store_true", help="Execute generated VMD script if vmd is available"
    )
    ap.add_argument("--vmd", default="", help="Path to vmd binary (optional)")
    args = ap.parse_args()

    outdir = str(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    script_path = os.path.join(outdir, "vmd_pipeline.tcl")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(
            "display projection Orthographic\n"
            f"mol new {{{os.path.abspath(args.traj)}}} type lammpstrj waitfor all\n"
            "mol modstyle 0 0 VDW 1.000000 12.000000\n"
            "mol modcolor 0 0 Type\n"
            "puts \"vmd_frames [molinfo top get numframes]\"\n"
            "quit\n"
        )

    vmd = str(args.vmd).strip() or shutil.which("vmd") or ""
    status = "prepared"
    rc = 0
    run_cmd = []
    stdout = ""
    stderr = ""
    if args.run:
        if not vmd:
            status = "skipped_missing_vmd"
        else:
            run_cmd = [vmd, "-dispdev", "text", "-e", script_path]
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
    out_json = os.path.join(outdir, "vmd_adapter.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
