"""TDMD-TD package.

Version is single-sourced from the repository root VERSION file.
"""

from __future__ import annotations
from pathlib import Path

def _read_version() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        return (repo_root / "VERSION").read_text(encoding="utf-8").strip()
    except Exception:
        return "4.5.0"

__version__ = _read_version()
