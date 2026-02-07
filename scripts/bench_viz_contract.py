#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True)
    return int(proc.returncode), str(proc.stdout or ""), str(proc.stderr or "")


def _write_csv(path: str, row: dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cols = list(row.keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow(row)


def _write_md(path: str, summary: dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Visualization Contract Smoke\n\n")
        f.write(f"- ok_all: `{bool(summary.get('ok_all', False))}`\n")
        f.write(f"- total: `{int(summary.get('total', 0))}`\n")
        f.write(f"- ok: `{int(summary.get('ok', 0))}`\n")
        f.write(f"- fail: `{int(summary.get('fail', 0))}`\n")
        f.write("\n## Artifacts\n\n")
        for k, v in dict(summary.get("artifacts", {}) or {}).items():
            f.write(f"- {k}: `{v}`\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualization contract smoke benchmark")
    ap.add_argument("--profile", default="", help="Reserved for contract compatibility")
    ap.add_argument("--out", required=True, help="CSV output path")
    ap.add_argument("--md", required=True, help="Markdown output path")
    ap.add_argument("--json", required=True, help="JSON output path")
    ap.add_argument("--strict", action="store_true", help="Fail on any contract violation")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out)) or os.getcwd()
    viz_dir = os.path.join(out_dir, "viz_contract")
    os.makedirs(viz_dir, exist_ok=True)

    traj = os.path.join(viz_dir, "traj.lammpstrj.gz")
    metrics = os.path.join(viz_dir, "metrics.csv")
    cmd_run = [
        sys.executable,
        "-m",
        "tdmd.main",
        "run",
        "--task",
        "examples/interop/task.yaml",
        "--mode",
        "serial",
        "--device",
        "cpu",
        "--traj",
        traj,
        "--traj-every",
        "1",
        "--traj-channels",
        "unwrapped,image",
        "--traj-compression",
        "gz",
        "--metrics",
        metrics,
        "--metrics-every",
        "1",
    ]
    run_rc, run_out, run_err = _run(cmd_run)

    bundle_dir = os.path.join(viz_dir, "bundle")
    cmd_bundle = [
        sys.executable,
        "scripts/viz_bundle.py",
        "--traj",
        traj,
        "--outdir",
        bundle_dir,
    ]
    bundle_rc, bundle_out, bundle_err = _run(cmd_bundle)

    required = [
        traj,
        f"{traj}.manifest.json",
        metrics,
        f"{metrics}.manifest.json",
        os.path.join(bundle_dir, "analysis.csv"),
        os.path.join(bundle_dir, "analysis.json"),
        os.path.join(bundle_dir, "ovito", "ovito_adapter.json"),
        os.path.join(bundle_dir, "vmd", "vmd_adapter.json"),
        os.path.join(bundle_dir, "paraview", "paraview_adapter.json"),
        os.path.join(bundle_dir, "viz_bundle.json"),
    ]
    missing = [p for p in required if not os.path.exists(p)]

    ok = bool(run_rc == 0 and bundle_rc == 0 and len(missing) == 0)
    row = {
        "case": "viz_contract_smoke",
        "ok": int(ok),
        "run_returncode": int(run_rc),
        "bundle_returncode": int(bundle_rc),
        "missing_artifacts": int(len(missing)),
    }
    _write_csv(args.out, row)

    summary = {
        "n": 1,
        "total": 1,
        "ok": int(ok),
        "fail": int(not ok),
        "ok_all": bool(ok),
        "worst": {
            "missing_artifacts": int(len(missing)),
            "run_returncode": int(run_rc),
            "bundle_returncode": int(bundle_rc),
        },
        "by_case": {
            "viz_contract_smoke": {
                "total": 1,
                "ok": int(ok),
                "fail": int(not ok),
                "worst": {
                    "missing_artifacts": int(len(missing)),
                },
            }
        },
        "rows": [row],
        "artifacts": {
            "traj": traj,
            "metrics": metrics,
            "bundle_dir": bundle_dir,
            "out_csv": args.out,
            "out_md": args.md,
            "out_json": args.json,
        },
        "missing": missing,
        "commands": {
            "run": cmd_run,
            "bundle": cmd_bundle,
        },
        "stdout": {
            "run": run_out,
            "bundle": bundle_out,
        },
        "stderr": {
            "run": run_err,
            "bundle": bundle_err,
        },
    }
    _write_md(args.md, summary)
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.strict and not ok:
        raise SystemExit(2)
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()

