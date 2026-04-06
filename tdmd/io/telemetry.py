from __future__ import annotations

import json
import os
import resource
import sys
import threading
import time
import warnings
from typing import Any

import numpy as np

from .manifest import telemetry_manifest_payload, write_manifest


def _mass_array_or_scalar(mass: float | np.ndarray) -> float | np.ndarray:
    if np.isscalar(mass):
        return float(mass)
    return np.asarray(mass, dtype=float)


def _temperature_from_velocity(v: np.ndarray, mass: float | np.ndarray) -> float:
    vv = np.asarray(v, dtype=float)
    n_atoms = int(vv.shape[0])
    if n_atoms <= 0:
        return 0.0
    if np.isscalar(mass):
        ke = 0.5 * float(mass) * float((vv * vv).sum())
    else:
        masses = np.asarray(mass, dtype=float)
        if masses.ndim != 1 or masses.shape[0] != n_atoms:
            return 0.0
        ke = 0.5 * float((masses[:, None] * (vv * vv)).sum())
    return (2.0 * ke) / (3.0 * max(1, n_atoms))


def _read_proc_status_value_mb(key: str) -> float | None:
    status_path = "/proc/self/status"
    if not os.path.exists(status_path):
        return None
    try:
        with open(status_path, encoding="utf-8") as f:
            for line in f:
                if not line.startswith(f"{key}:"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    return None
                return float(parts[1]) / 1024.0
    except OSError:
        return None
    return None


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    raw = float(getattr(usage, "ru_maxrss", 0.0))
    if sys.platform == "darwin":
        return raw / (1024.0 * 1024.0)
    return raw / 1024.0


def _gpu_snapshot(device: str) -> dict[str, Any]:
    if str(device) != "cuda":
        return {}
    try:
        import cupy as cp  # noqa: F401
    except Exception:
        return {}
    try:
        dev_id = int(cp.cuda.runtime.getDevice())
        props = cp.cuda.runtime.getDeviceProperties(dev_id)
        name = props.get("name", b"")
        if isinstance(name, bytes):
            gpu_name = name.decode("utf-8", errors="replace")
        else:
            gpu_name = str(name)
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        pool = cp.get_default_memory_pool()
        pinned_pool = cp.get_default_pinned_memory_pool()
        return {
            "gpu_device_id": dev_id,
            "gpu_name": gpu_name,
            "gpu_device_free_mb": float(free_bytes) / (1024.0 * 1024.0),
            "gpu_device_used_mb": float(total_bytes - free_bytes) / (1024.0 * 1024.0),
            "gpu_device_total_mb": float(total_bytes) / (1024.0 * 1024.0),
            "gpu_pool_used_mb": float(pool.used_bytes()) / (1024.0 * 1024.0),
            "gpu_pool_total_mb": float(pool.total_bytes()) / (1024.0 * 1024.0),
            "gpu_pinned_pool_free_blocks": int(pinned_pool.n_free_blocks()),
        }
    except Exception:
        return {}


class TelemetryWriter:
    SCHEMA_NAME = "tdmd.telemetry.jsonl"
    SCHEMA_VERSION = 1

    def __init__(
        self,
        path: str | None,
        *,
        total_steps: int,
        mass: float | np.ndarray,
        atom_count: int,
        device: str,
        mode: str,
        write_output_manifest: bool = True,
        emit_stdout: bool = False,
        heartbeat_every_sec: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ):
        self.path = str(path) if path else ""
        self.total_steps = max(0, int(total_steps))
        self.mass = _mass_array_or_scalar(mass)
        self.atom_count = int(atom_count)
        self.device = str(device)
        self.mode = str(mode)
        self.emit_stdout = bool(emit_stdout)
        self.heartbeat_every_sec = max(0.0, float(heartbeat_every_sec))
        self.metadata = dict(metadata or {})
        self._created = time.perf_counter()
        self._last_wall_sec = 0.0
        self._last_step = 0
        self._count = 0
        self._closed = False
        self._last_record: dict[str, Any] | None = None
        self._max_rss_mb = 0.0
        self._max_gpu_device_used_mb = 0.0
        self._max_gpu_pool_used_mb = 0.0
        self._last_model_state: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._field_names = [
            "record_kind",
            "step",
            "steps_total",
            "step_fraction",
            "wall_sec",
            "delta_wall_sec",
            "steps_per_sec_avg",
            "steps_per_sec_inst",
            "eta_sec",
            "atom_count",
            "box",
            "coord_min",
            "coord_max",
            "speed_mean",
            "vmax",
            "temperature_est",
            "nonfinite_r",
            "nonfinite_v",
            "buffer",
            "rss_mb",
            "peak_rss_mb",
            "cpu_user_sec",
            "cpu_system_sec",
            "device",
            "mode",
            "gpu_device_id",
            "gpu_name",
            "gpu_device_free_mb",
            "gpu_device_used_mb",
            "gpu_device_total_mb",
            "gpu_pool_used_mb",
            "gpu_pool_total_mb",
            "gpu_pinned_pool_free_blocks",
        ]
        self._f = None
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            self._f = open(self.path, "w", encoding="utf-8")
            if write_output_manifest:
                mpath = f"{self.path}.manifest.json"
                payload = telemetry_manifest_payload(
                    path=self.path,
                    format_name=self.SCHEMA_NAME,
                    schema_version=self.SCHEMA_VERSION,
                    fields=list(self._field_names),
                    atom_count=self.atom_count,
                    total_steps=self.total_steps,
                    device=self.device,
                    mode=self.mode,
                    metadata=self.metadata,
                )
                write_manifest(mpath, payload)
        if self.heartbeat_every_sec > 0.0:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name="tdmd-telemetry-heartbeat",
                daemon=True,
            )
            self._heartbeat_thread.start()

    @property
    def last_record(self) -> dict[str, Any] | None:
        return None if self._last_record is None else dict(self._last_record)

    @property
    def summary_path(self) -> str:
        return f"{self.path}.summary.json" if self.path else ""

    def _observe_model_state(
        self,
        *,
        r: np.ndarray,
        v: np.ndarray,
        box_value: float | None,
        buffer_value: float,
    ) -> dict[str, Any]:
        rr = np.asarray(r, dtype=float)
        vv = np.asarray(v, dtype=float)
        speeds = np.linalg.norm(vv, axis=1) if vv.size else np.zeros((0,), dtype=float)
        coord_min = rr.min(axis=0).tolist() if rr.size else [0.0, 0.0, 0.0]
        coord_max = rr.max(axis=0).tolist() if rr.size else [0.0, 0.0, 0.0]
        return {
            "atom_count": int(rr.shape[0]),
            "box": None if box_value is None else float(box_value),
            "coord_min": [float(x) for x in coord_min],
            "coord_max": [float(x) for x in coord_max],
            "speed_mean": float(speeds.mean()) if speeds.size else 0.0,
            "vmax": float(speeds.max()) if speeds.size else 0.0,
            "temperature_est": float(_temperature_from_velocity(vv, self.mass)),
            "nonfinite_r": int(np.size(rr) - int(np.isfinite(rr).sum())),
            "nonfinite_v": int(np.size(vv) - int(np.isfinite(vv).sum())),
            "buffer": float(buffer_value),
        }

    def _make_record(
        self,
        *,
        record_kind: str,
        step_i: int,
        delta_steps: int,
        model_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        wall_sec = float(time.perf_counter() - self._created)
        delta_wall_sec = float(max(0.0, wall_sec - self._last_wall_sec))
        self._last_wall_sec = wall_sec
        total = int(self.total_steps)
        steps_per_sec_avg = (float(step_i) / wall_sec) if wall_sec > 0.0 and step_i > 0 else 0.0
        steps_per_sec_inst = (
            float(max(1, delta_steps)) / delta_wall_sec
            if delta_wall_sec > 0.0 and delta_steps > 0
            else steps_per_sec_avg
        )
        eta_sec = (
            float(max(0, total - step_i)) / steps_per_sec_avg
            if total > 0 and steps_per_sec_avg > 0.0
            else None
        )
        rss_mb = _read_proc_status_value_mb("VmRSS")
        peak_rss_mb = _read_proc_status_value_mb("VmHWM")
        if rss_mb is None:
            rss_mb = _peak_rss_mb()
        if peak_rss_mb is None:
            peak_rss_mb = _peak_rss_mb()
        proc_times = os.times()
        gpu = _gpu_snapshot(self.device)
        record = {
            "record_kind": str(record_kind),
            "step": step_i,
            "steps_total": total,
            "step_fraction": (float(step_i) / float(total)) if total > 0 else 0.0,
            "wall_sec": wall_sec,
            "delta_wall_sec": delta_wall_sec,
            "steps_per_sec_avg": steps_per_sec_avg,
            "steps_per_sec_inst": steps_per_sec_inst,
            "eta_sec": eta_sec,
            "device": self.device,
            "mode": self.mode,
            "rss_mb": float(rss_mb),
            "peak_rss_mb": float(peak_rss_mb),
            "cpu_user_sec": float(proc_times.user),
            "cpu_system_sec": float(proc_times.system),
        }
        if model_state is not None:
            record.update(model_state)
        else:
            record.update(
                {
                    "atom_count": self.atom_count,
                    "box": None,
                    "coord_min": [0.0, 0.0, 0.0],
                    "coord_max": [0.0, 0.0, 0.0],
                    "speed_mean": 0.0,
                    "vmax": 0.0,
                    "temperature_est": 0.0,
                    "nonfinite_r": 0,
                    "nonfinite_v": 0,
                    "buffer": 0.0,
                }
            )
        record.update(gpu)
        return record

    def _persist_record(self, record: dict[str, Any]) -> dict[str, Any]:
        self._last_record = record
        self._count += 1
        self._max_rss_mb = max(self._max_rss_mb, float(record["rss_mb"]))
        self._max_gpu_device_used_mb = max(
            self._max_gpu_device_used_mb,
            float(record.get("gpu_device_used_mb", 0.0) or 0.0),
        )
        self._max_gpu_pool_used_mb = max(
            self._max_gpu_pool_used_mb,
            float(record.get("gpu_pool_used_mb", 0.0) or 0.0),
        )
        if self._f is not None:
            self._f.write(json.dumps(record, sort_keys=True) + "\n")
            self._f.flush()
        if self.emit_stdout:
            eta_txt = "n/a" if record.get("eta_sec") is None else f"{float(record['eta_sec']):.1f}s"
            gpu_used = record.get("gpu_device_used_mb")
            gpu_txt = "" if gpu_used is None else f" gpu_used={float(gpu_used):.1f}MB"
            print(
                "[telemetry] "
                f"kind={record['record_kind']} mode={self.mode} device={self.device} "
                f"step={int(record['step'])}/{self.total_steps} "
                f"wall={float(record['wall_sec']):.1f}s avg={float(record['steps_per_sec_avg']):.3f}step/s "
                f"eta={eta_txt} rss={float(record['rss_mb']):.1f}MB "
                f"vmax={float(record['vmax']):.4f} T={float(record['temperature_est']):.4f}{gpu_txt}",
                flush=True,
            )
        return dict(record)

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.heartbeat_every_sec):
            with self._lock:
                if self._closed:
                    return
                record = self._make_record(
                    record_kind="heartbeat",
                    step_i=int(self._last_step),
                    delta_steps=0,
                    model_state=None if self._last_model_state is None else dict(self._last_model_state),
                )
                self._persist_record(record)

    def __call__(self, step: int, r: np.ndarray, v: np.ndarray, box: float | None = None) -> None:
        self.write(step, r, v, box_value=box, buffer_value=0.0)

    def write(
        self,
        step: int,
        r: np.ndarray,
        v: np.ndarray,
        *,
        box_value: float | None = None,
        buffer_value: float = 0.0,
    ) -> dict[str, Any]:
        with self._lock:
            step_i = int(step)
            delta_steps = int(step_i - self._last_step) if self._count > 0 else int(step_i)
            self._last_step = step_i
            self._last_model_state = self._observe_model_state(
                r=r,
                v=v,
                box_value=box_value,
                buffer_value=buffer_value,
            )
            record = self._make_record(
                record_kind="observer",
                step_i=step_i,
                delta_steps=delta_steps,
                model_state=dict(self._last_model_state),
            )
            return self._persist_record(record)

    def close(self, *, completed: bool | None = None) -> dict[str, Any] | None:
        with self._lock:
            if self._closed:
                return self.last_record
            self._closed = True
            self._stop_event.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=max(1.0, self.heartbeat_every_sec * 2.0))
        with self._lock:
            summary = {
                "schema": {
                    "name": self.SCHEMA_NAME,
                    "version": self.SCHEMA_VERSION,
                },
                "path": self.path or None,
                "summary_path": self.summary_path or None,
                "device": self.device,
                "mode": self.mode,
                "atom_count": self.atom_count,
                "steps_total": self.total_steps,
                "records": int(self._count),
                "completed": (
                    bool(completed)
                    if completed is not None
                    else bool(self._last_record and int(self._last_record.get("step", -1)) >= self.total_steps)
                ),
                "wall_sec": float(time.perf_counter() - self._created),
                "max_rss_mb": float(self._max_rss_mb),
                "max_gpu_device_used_mb": float(self._max_gpu_device_used_mb),
                "max_gpu_pool_used_mb": float(self._max_gpu_pool_used_mb),
                "metadata": dict(self.metadata),
                "last_record": self.last_record,
            }
            if self._f is not None:
                try:
                    self._f.close()
                except OSError as exc:
                    warnings.warn(
                        f"TelemetryWriter.close() failed for {self.path!r}: {exc!r}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                try:
                    with open(self.summary_path, "w", encoding="utf-8") as f:
                        json.dump(summary, f, indent=2, sort_keys=True)
                except OSError as exc:
                    warnings.warn(
                        f"TelemetryWriter summary write failed for {self.summary_path!r}: {exc!r}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            return self.last_record
