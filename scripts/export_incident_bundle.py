from __future__ import annotations

import argparse
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tdmd.incident_bundle import export_incident_bundle_zip, write_incident_bundle


def main() -> int:
    p = argparse.ArgumentParser(
        description="Export TDMD incident bundle from a VerifyLab run directory"
    )
    p.add_argument("--run-dir", required=True, help="Run directory (e.g. results/<run_id>)")
    p.add_argument("--reason", default="manual_export", help="Bundle reason label")
    p.add_argument(
        "--bundle-name",
        default="incident_bundle_manual",
        help="Bundle directory name under run-dir",
    )
    p.add_argument("--zip-out", default="", help="Optional zip output path")
    args = p.parse_args()

    run_dir = os.path.abspath(str(args.run_dir))
    if not os.path.isdir(run_dir):
        print(f"[incident-bundle] run-dir not found: {run_dir}", file=sys.stderr)
        return 2

    bundle_dir = write_incident_bundle(
        run_dir=run_dir,
        reason=str(args.reason),
        extra={"manual_export": True},
        bundle_name=str(args.bundle_name),
    )
    print(f"[incident-bundle] bundle_dir={bundle_dir}", flush=True)
    if str(args.zip_out).strip():
        zip_path = export_incident_bundle_zip(bundle_dir=bundle_dir, out_zip=str(args.zip_out))
        print(f"[incident-bundle] zip={zip_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
