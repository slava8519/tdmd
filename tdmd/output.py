from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io.metrics import MetricsWriter
from .io.telemetry import TelemetryWriter
from .io.trajectory import TrajectoryWriter


@dataclass
class OutputBundle:
    traj: TrajectoryWriter | None = None
    metrics: MetricsWriter | None = None
    telemetry: TelemetryWriter | None = None

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
        if self.telemetry is not None:
            self.telemetry.write(step, r, v, buffer_value=buffer_value, box_value=box_value)

    def close(self):
        if self.traj is not None:
            self.traj.close()
        if self.metrics is not None:
            self.metrics.close()
        if self.telemetry is not None:
            self.telemetry.close()


@dataclass(frozen=True)
class OutputSpec:
    traj_path: str | None
    traj_every: int
    metrics_path: str | None
    metrics_every: int
    atom_ids: np.ndarray
    atom_types: np.ndarray
    box: tuple[float, float, float]
    pbc: tuple[bool, bool, bool]
    mass: float | np.ndarray
    cutoff: float
    potential: object
    traj_channels: tuple[str, ...] = ()
    traj_compression: str = "none"
    write_output_manifest: bool = True
    telemetry_path: str | None = None
    telemetry_every: int = 0
    telemetry_stdout: bool = False
    telemetry_heartbeat_sec: float = 0.0
    total_steps: int = 0
    device: str = "cpu"
    mode: str = "serial"


def make_output_bundle(spec: OutputSpec | None) -> OutputBundle:
    """Create output writers from an OutputSpec.

    Note: cadence (every/period) is enforced by the caller.
    Writers are only created if the corresponding path is set and `*_every > 0`.
    """
    if spec is None:
        return OutputBundle(traj=None, metrics=None, telemetry=None)

    traj_writer = None
    metrics_writer = None
    telemetry_writer = None

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
    if (spec.telemetry_path or spec.telemetry_stdout) and int(spec.telemetry_every) > 0:
        telemetry_writer = TelemetryWriter(
            spec.telemetry_path,
            total_steps=int(spec.total_steps),
            mass=spec.mass,
            atom_count=int(spec.atom_types.shape[0]),
            device=str(spec.device),
            mode=str(spec.mode),
            write_output_manifest=bool(spec.write_output_manifest),
            emit_stdout=bool(spec.telemetry_stdout),
            heartbeat_every_sec=float(spec.telemetry_heartbeat_sec),
            metadata={
                "traj_path": spec.traj_path,
                "metrics_path": spec.metrics_path,
            },
        )

    return OutputBundle(traj=traj_writer, metrics=metrics_writer, telemetry=telemetry_writer)
