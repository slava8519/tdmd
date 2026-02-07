#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tdmd.viz import build_plugin, run_plugins


def _write_rows_csv(path: str, rows: list[dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step"])
        return
    cols = sorted({k for r in rows for k in r.keys()})
    # Keep step first for readability.
    if "step" in cols:
        cols = ["step"] + [c for c in cols if c != "step"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Universal trajectory post-processing (plugin-based)")
    ap.add_argument("--traj", required=True, help="Trajectory path (.lammpstrj or .lammpstrj.gz)")
    ap.add_argument(
        "--plugin",
        action="append",
        default=[],
        help="Plugin spec; repeatable. Example: mobility | species_mixing | region:xlo=0,xhi=10,ylo=0,yhi=10,zlo=0,zhi=10",
    )
    ap.add_argument("--every", type=int, default=1, help="Frame stride")
    ap.add_argument("--start-step", type=int, default=None, help="Optional minimum step")
    ap.add_argument("--stop-step", type=int, default=None, help="Optional maximum step")
    ap.add_argument("--out-csv", default="", help="Output per-frame CSV")
    ap.add_argument("--out-json", default="", help="Output JSON summary")
    args = ap.parse_args()

    specs = list(args.plugin or [])
    if not specs:
        specs = ["mobility", "species_mixing"]
    plugins = [build_plugin(x) for x in specs]
    rows, summary = run_plugins(
        args.traj,
        plugins,
        every=max(1, int(args.every)),
        start_step=args.start_step,
        stop_step=args.stop_step,
    )
    payload = {
        "ok": True,
        "traj": str(args.traj),
        "plugins": specs,
        "rows": len(rows),
        "summary": summary,
    }
    if args.out_csv:
        _write_rows_csv(args.out_csv, rows)
        payload["out_csv"] = str(args.out_csv)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        payload["out_json"] = str(args.out_json)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

