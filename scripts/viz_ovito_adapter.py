#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare/run OVITO adapter artifacts")
    ap.add_argument("--traj", required=True, help="Input trajectory path")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--run", action="store_true", help="Execute generated OVITO script if ovitos is available")
    ap.add_argument("--ovitos", default="", help="Path to ovitos binary (optional)")
    args = ap.parse_args()

    outdir = str(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    script_path = os.path.join(outdir, "ovito_pipeline.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(
            "from ovito.io import import_file\n"
            "from ovito.modifiers import CommonNeighborAnalysisModifier\n\n"
            f"pipeline = import_file(r'''{os.path.abspath(args.traj)}''')\n"
            "pipeline.modifiers.append(CommonNeighborAnalysisModifier())\n"
            "# Example headless post-processing point:\n"
            "data = pipeline.compute()\n"
            "print('ovito_frames', pipeline.source.num_frames)\n"
            "print('ovito_particles', data.particles.count)\n"
        )

    ovitos = str(args.ovitos).strip() or shutil.which("ovitos") or ""
    status = "prepared"
    rc = 0
    run_cmd = []
    stdout = ""
    stderr = ""
    if args.run:
        if not ovitos:
            status = "skipped_missing_ovitos"
        else:
            run_cmd = [ovitos, script_path]
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
    out_json = os.path.join(outdir, "ovito_adapter.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

