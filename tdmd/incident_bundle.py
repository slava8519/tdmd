from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
import shutil
import zipfile
from typing import Any


_CORE_FILES = (
    "config.json",
    "summary.json",
    "summary.md",
    "metrics.csv",
)

_OPTIONAL_PREFIXES = (
    "mpi_overlap_",
    "td_trace",
)

_OPTIONAL_SUFFIXES = (
    ".log",
    ".txt",
)


@dataclass(frozen=True)
class BundleFile:
    relative_path: str
    size_bytes: int
    sha256: str


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _collect_source_files(run_dir: str) -> list[str]:
    out: list[str] = []
    for nm in _CORE_FILES:
        p = os.path.join(run_dir, nm)
        if os.path.isfile(p):
            out.append(p)
    for nm in sorted(os.listdir(run_dir)):
        p = os.path.join(run_dir, nm)
        if not os.path.isfile(p):
            continue
        if any(nm.startswith(pref) for pref in _OPTIONAL_PREFIXES):
            out.append(p)
            continue
        if any(nm.endswith(suf) for suf in _OPTIONAL_SUFFIXES):
            out.append(p)
            continue
    # keep deterministic order and remove duplicates while preserving first occurrence
    uniq: list[str] = []
    seen: set[str] = set()
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def write_incident_bundle(
    run_dir: str,
    *,
    reason: str,
    extra: dict[str, Any] | None = None,
    bundle_name: str = "incident_bundle",
) -> str:
    run_dir_abs = os.path.abspath(run_dir)
    bundle_dir = os.path.join(run_dir_abs, bundle_name)
    artifacts_dir = os.path.join(bundle_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    files = _collect_source_files(run_dir_abs)
    manifest_files: list[BundleFile] = []
    for src in files:
        dst = os.path.join(artifacts_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        manifest_files.append(
            BundleFile(
                relative_path=os.path.relpath(dst, bundle_dir),
                size_bytes=int(os.path.getsize(dst)),
                sha256=_sha256_file(dst),
            )
        )

    manifest = {
        "bundle_schema_version": 1,
        "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_dir": run_dir_abs,
        "reason": str(reason),
        "files": [x.__dict__ for x in manifest_files],
        "extra": dict(extra or {}),
    }
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    readme_path = os.path.join(bundle_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("TDMD incident bundle\n")
        f.write(f"reason: {reason}\n")
        f.write(f"run_dir: {run_dir_abs}\n")
        f.write(f"created_utc: {manifest['created_utc']}\n")
        f.write("files:\n")
        for ff in manifest_files:
            f.write(f"- {ff.relative_path} ({ff.size_bytes} bytes)\n")

    return bundle_dir


def export_incident_bundle_zip(bundle_dir: str, out_zip: str) -> str:
    bundle_dir_abs = os.path.abspath(bundle_dir)
    out_zip_abs = os.path.abspath(out_zip)
    os.makedirs(os.path.dirname(out_zip_abs) or ".", exist_ok=True)
    with zipfile.ZipFile(out_zip_abs, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(bundle_dir_abs):
            for nm in sorted(files):
                src = os.path.join(root, nm)
                arc = os.path.relpath(src, bundle_dir_abs)
                zf.write(src, arc)
    return out_zip_abs
