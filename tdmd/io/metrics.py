from __future__ import annotations
from typing import Union
import csv
import os
import numpy as np

from ..observables import compute_observables
from .manifest import metrics_manifest_payload, write_manifest

class MetricsWriter:
    SCHEMA_NAME = "tdmd.metrics.csv"
    SCHEMA_VERSION = 1

    def __init__(
        self,
        path: str,
        *,
        mass: Union[float, np.ndarray],
        box: float,
        cutoff: float,
        potential,
        atom_types: np.ndarray | None = None,
        write_output_manifest: bool = True,
    ):
        self.path = path
        self.mass = np.asarray(mass, dtype=float) if not np.isscalar(mass) else float(mass)
        self.atom_types = None if atom_types is None else np.asarray(atom_types, dtype=np.int32)
        self.box = float(box)
        self.cutoff = float(cutoff)
        self.potential = potential
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._columns = ["step","T","E_kin","E_pot","P","vmax","buffer"]
        self._w.writerow(self._columns)
        self._f.flush()
        if write_output_manifest:
            mpath = f"{self.path}.manifest.json"
            atom_count = int(self.atom_types.shape[0]) if self.atom_types is not None else 0
            mp = metrics_manifest_payload(
                path=self.path,
                format_name=self.SCHEMA_NAME,
                schema_version=self.SCHEMA_VERSION,
                columns=list(self._columns),
                cutoff=float(self.cutoff),
                atom_count=atom_count,
            )
            write_manifest(mpath, mp)

    def write(self, step: int, r: np.ndarray, v: np.ndarray, buffer_value: float = 0.0, box_value: float | None = None):
        box_cur = float(self.box if box_value is None else box_value)
        obs = compute_observables(
            r, v, self.mass, box_cur, self.potential, self.cutoff, atom_types=self.atom_types
        )
        speeds = np.linalg.norm(v, axis=1)
        vmax = float(speeds.max()) if speeds.size else 0.0
        self._w.writerow([
            int(step),
            float(obs.get("T", 0.0)),
            float(obs.get("KE", 0.0)),
            float(obs.get("PE", 0.0)),
            float(obs.get("P", 0.0)),
            vmax,
            float(buffer_value),
        ])
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass
