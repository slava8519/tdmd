from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from .io.metrics import MetricsWriter
from .io.trajectory import TrajectoryWriter


@dataclass
class OutputBundle:
    traj: Optional[TrajectoryWriter] = None
    metrics: Optional[MetricsWriter] = None

    def on_step(
        self,
        step: int,
        r: np.ndarray,
        v: np.ndarray,
        buffer_value: float = 0.0,
        box_value: float | None = None,
    ):
        if self.traj is not None:
            b = (
                None
                if box_value is None
                else (float(box_value), float(box_value), float(box_value))
            )
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


def make_output_bundle(spec: OutputSpec | None) -> OutputBundle:
    """Create output writers from an OutputSpec.

    Note: cadence (every/period) is enforced by the caller.
    Writers are only created if the corresponding path is set and `*_every > 0`.
    """
    if spec is None:
        return OutputBundle(traj=None, metrics=None)

    traj_writer = None
    metrics_writer = None

    if spec.traj_path and int(spec.traj_every) > 0:
        traj_writer = TrajectoryWriter(
            spec.traj_path,
            box=spec.box,
            pbc=spec.pbc,
            atom_ids=spec.atom_ids,
            atom_types=spec.atom_types,
            channels=tuple(spec.traj_channels or ()),
            compression=str(spec.traj_compression),
            write_output_manifest=bool(spec.write_output_manifest),
            force_potential=spec.potential,
            force_cutoff=float(spec.cutoff),
            force_atom_types=spec.atom_types,
        )
    if spec.metrics_path and int(spec.metrics_every) > 0:
        metrics_writer = MetricsWriter(
            spec.metrics_path,
            mass=spec.mass,
            box=float(spec.box[0]),
            cutoff=float(spec.cutoff),
            potential=spec.potential,
            atom_types=spec.atom_types,
            write_output_manifest=bool(spec.write_output_manifest),
        )

    return OutputBundle(traj=traj_writer, metrics=metrics_writer)
