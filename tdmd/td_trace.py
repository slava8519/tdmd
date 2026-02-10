from __future__ import annotations
import csv
import os
import time
import warnings
from typing import Optional

from .zones import ZoneType

def _state_label(z) -> str:
    if isinstance(z, ZoneType):
        return z.name
    if isinstance(z, str):
        return z.upper()
    return str(z)

def format_invariant_flags(diag: dict) -> str:
    keys = []
    for k, v in diag.items():
        try:
            if float(v) != 0.0:
                keys.append(str(k))
        except Exception:
            continue
    return ",".join(sorted(keys))

class TDTraceLogger:
    def __init__(self, path: str, *, rank: int, enabled: bool = True):
        self.enabled = bool(enabled)
        self.rank = int(rank)
        self.start = time.perf_counter()
        self.path = path
        if not self.enabled:
            self._f = None
            self._w = None
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow([
            "wall_time","rank","step_id","zone_id","event","state_before","state_after",
            "halo_ids_count","migration_count","lag","invariant_flags",
        ])
        self._f.flush()

    def log(self, *, step_id: int, zone_id: int, event: str,
            state_before: Optional[str] = None, state_after: Optional[str] = None,
            halo_ids_count: int = 0, migration_count: int = 0, lag: int = 0,
            invariant_flags: str = ""):
        if not self.enabled or self._w is None:
            return
        wall = time.perf_counter() - self.start
        self._w.writerow([
            f"{wall:.6f}",
            int(self.rank),
            int(step_id),
            int(zone_id),
            str(event),
            _state_label(state_before) if state_before is not None else "",
            _state_label(state_after) if state_after is not None else "",
            int(halo_ids_count),
            int(migration_count),
            int(lag),
            str(invariant_flags),
        ])
        self._f.flush()

    def close(self):
        try:
            if self._f is not None:
                self._f.close()
        except Exception as exc:
            warnings.warn(
                f"TDTraceLogger.close() failed for {self.path!r}: {exc!r}",
                RuntimeWarning,
            )
