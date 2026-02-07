from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np

from .io.trajectory import TrajectoryWriter
from .io.metrics import MetricsWriter

@dataclass
class OutputBundle:
    traj: Optional[TrajectoryWriter] = None
    metrics: Optional[MetricsWriter] = None

    def on_step(self, step: int, r: np.ndarray, v: np.ndarray, buffer_value: float = 0.0, box_value: float | None = None):
        if self.traj is not None:
            b = None if box_value is None else (float(box_value), float(box_value), float(box_value))
            self.traj.write(step, r, v, box_value=b)
        if self.metrics is not None:
            self.metrics.write(step, r, v, buffer_value=buffer_value, box_value=box_value)

    def close(self):
        if self.traj is not None:
            self.traj.close()
        if self.metrics is not None:
            self.metrics.close()

@dataclass(frozen=True)
class OutputSpec:
    traj_path: Optional[str]
    traj_every: int
    metrics_path: Optional[str]
    metrics_every: int
    atom_ids: np.ndarray
    atom_types: np.ndarray
    box: tuple[float, float, float]
    pbc: tuple[bool, bool, bool]
    mass: Union[float, np.ndarray]
    cutoff: float
    potential: object
    traj_channels: tuple[str, ...] = ()
    traj_compression: str = "none"
    write_output_manifest: bool = True
