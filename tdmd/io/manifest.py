from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_manifest(path: str, payload: dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def trajectory_manifest_payload(
    *,
    path: str,
    format_name: str,
    schema_version: int,
    columns: list[str],
    n_atoms: int,
    pbc: tuple[bool, bool, bool],
    box: tuple[float, float, float],
    channels: dict[str, bool],
    compression: str,
) -> dict[str, Any]:
    return {
        "kind": "trajectory",
        "schema": {
            "name": str(format_name),
            "version": int(schema_version),
        },
        "created_at_utc": _utc_now_iso(),
        "path": str(path),
        "columns": list(columns),
        "n_atoms": int(n_atoms),
        "pbc": [bool(pbc[0]), bool(pbc[1]), bool(pbc[2])],
        "box": [float(box[0]), float(box[1]), float(box[2])],
        "channels": {str(k): bool(v) for k, v in channels.items()},
        "compression": str(compression),
    }


def metrics_manifest_payload(
    *,
    path: str,
    format_name: str,
    schema_version: int,
    columns: list[str],
    cutoff: float,
    atom_count: int,
) -> dict[str, Any]:
    return {
        "kind": "metrics",
        "schema": {
            "name": str(format_name),
            "version": int(schema_version),
        },
        "created_at_utc": _utc_now_iso(),
        "path": str(path),
        "columns": list(columns),
        "cutoff": float(cutoff),
        "atom_count": int(atom_count),
    }
