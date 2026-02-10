from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RuntimeProfile:
    config: str
    mpirun: str
    timeout_sec: int
    retries: int
    strict_invariants: bool
    strict_zone_width: bool
    overlap_list: list[int]
    ranks_default: list[int]
    ranks_per_node: int
    env: dict[str, str]
    allow_simulated_cluster: bool
    prefer_simulated: bool
    stability_steps: int
    stability_thermo_every: int


@dataclass
class ScalingProfile:
    strong_ranks: list[int]
    weak_ranks: list[int]
    overlap: int
    cuda_aware_mpi: bool
    min_speedup: float
    min_efficiency: float
    max_weak_elapsed_ratio: float


@dataclass
class StabilityProfile:
    ranks: list[int]
    overlap: int
    cuda_aware_mpi: bool
    max_hG: int
    max_hV: int
    max_violW: int
    max_lagV: int
    min_diag_samples: int


@dataclass
class TransportEntry:
    name: str
    overlap: int
    cuda_aware_mpi: bool
    fabric: str
    env: dict[str, str]


@dataclass
class TransportMatrixProfile:
    ranks: list[int]
    entries: list[TransportEntry]


@dataclass
class ClusterProfile:
    profile_version: int
    name: str
    description: str
    runtime: RuntimeProfile
    scaling: ScalingProfile
    stability: StabilityProfile
    transport_matrix: TransportMatrixProfile
    metadata: dict[str, Any]
    source_path: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["source_path"] = self.source_path
        return d


def _as_int_list(values: Any, *, default: list[int]) -> list[int]:
    if values is None:
        return list(default)
    out: list[int] = []
    for x in list(values):
        out.append(int(x))
    return out


def _as_overlap_list(values: Any, *, default: list[int]) -> list[int]:
    vals = _as_int_list(values, default=default)
    if not vals:
        raise ValueError("overlap list must not be empty")
    for v in vals:
        if int(v) not in (0, 1):
            raise ValueError("overlap values must be 0 or 1")
    return vals


def _normalize_env(env: Any) -> dict[str, str]:
    if not isinstance(env, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in env.items():
        if not str(k).strip():
            continue
        out[str(k)] = str(v)
    return out


def load_cluster_profile(path: str) -> ClusterProfile:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"cluster profile not found: {path}")
    with src.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("cluster profile root must be a mapping")
    if int(data.get("profile_version", 0)) != 1:
        raise ValueError("cluster profile: profile_version=1 is required")

    runtime_d = dict(data.get("runtime", {}) or {})
    scaling_d = dict(data.get("scaling", {}) or {})
    stability_d = dict(data.get("stability", {}) or {})
    matrix_d = dict(data.get("transport_matrix", {}) or {})
    entries_d = list(matrix_d.get("profiles", []) or [])

    runtime = RuntimeProfile(
        config=str(runtime_d.get("config", "examples/td_1d_morse_static_rr_smoke4.yaml")),
        mpirun=str(runtime_d.get("mpirun", "")).strip(),
        timeout_sec=int(runtime_d.get("timeout_sec", 180)),
        retries=max(1, int(runtime_d.get("retries", 2))),
        strict_invariants=bool(runtime_d.get("strict_invariants", True)),
        strict_zone_width=bool(runtime_d.get("strict_zone_width", True)),
        overlap_list=_as_overlap_list(runtime_d.get("overlap_list"), default=[0, 1]),
        ranks_default=_as_int_list(runtime_d.get("ranks_default"), default=[2, 4]),
        ranks_per_node=max(1, int(runtime_d.get("ranks_per_node", 2))),
        env=_normalize_env(runtime_d.get("env", {})),
        allow_simulated_cluster=bool(runtime_d.get("allow_simulated_cluster", True)),
        prefer_simulated=bool(runtime_d.get("prefer_simulated", True)),
        stability_steps=max(1, int(runtime_d.get("stability_steps", 80))),
        stability_thermo_every=max(1, int(runtime_d.get("stability_thermo_every", 1))),
    )
    if not runtime.ranks_default:
        raise ValueError("runtime.ranks_default must not be empty")

    scaling = ScalingProfile(
        strong_ranks=_as_int_list(scaling_d.get("strong_ranks"), default=runtime.ranks_default),
        weak_ranks=_as_int_list(scaling_d.get("weak_ranks"), default=runtime.ranks_default),
        overlap=int(scaling_d.get("overlap", 1)),
        cuda_aware_mpi=bool(scaling_d.get("cuda_aware_mpi", False)),
        min_speedup=float(scaling_d.get("min_speedup", 0.4)),
        min_efficiency=float(scaling_d.get("min_efficiency", 0.15)),
        max_weak_elapsed_ratio=float(scaling_d.get("max_weak_elapsed_ratio", 2.5)),
    )
    if scaling.overlap not in (0, 1):
        raise ValueError("scaling.overlap must be 0 or 1")
    if not scaling.strong_ranks or not scaling.weak_ranks:
        raise ValueError("scaling rank lists must not be empty")

    stability = StabilityProfile(
        ranks=_as_int_list(stability_d.get("ranks"), default=runtime.ranks_default),
        overlap=int(stability_d.get("overlap", 1)),
        cuda_aware_mpi=bool(stability_d.get("cuda_aware_mpi", False)),
        max_hG=max(0, int(stability_d.get("max_hG", 0))),
        max_hV=max(0, int(stability_d.get("max_hV", 0))),
        max_violW=max(0, int(stability_d.get("max_violW", 0))),
        max_lagV=max(0, int(stability_d.get("max_lagV", 0))),
        min_diag_samples=max(0, int(stability_d.get("min_diag_samples", 1))),
    )
    if stability.overlap not in (0, 1):
        raise ValueError("stability.overlap must be 0 or 1")
    if not stability.ranks:
        raise ValueError("stability.ranks must not be empty")

    entries: list[TransportEntry] = []
    for e in entries_d:
        if not isinstance(e, dict):
            continue
        ov = int(e.get("overlap", 0))
        if ov not in (0, 1):
            raise ValueError("transport_matrix.profiles[].overlap must be 0 or 1")
        entries.append(
            TransportEntry(
                name=str(e.get("name", f"profile_{len(entries)}")),
                overlap=ov,
                cuda_aware_mpi=bool(e.get("cuda_aware_mpi", False)),
                fabric=str(e.get("fabric", "unknown")),
                env=_normalize_env(e.get("env", {})),
            )
        )
    if not entries:
        entries = [
            TransportEntry(
                name="host_blocking", overlap=0, cuda_aware_mpi=False, fabric="host_staging", env={}
            ),
            TransportEntry(
                name="host_overlap", overlap=1, cuda_aware_mpi=False, fabric="host_staging", env={}
            ),
            TransportEntry(
                name="cudaaware_overlap",
                overlap=1,
                cuda_aware_mpi=True,
                fabric="cuda_aware",
                env={},
            ),
        ]
    matrix = TransportMatrixProfile(
        ranks=_as_int_list(matrix_d.get("ranks"), default=runtime.ranks_default),
        entries=entries,
    )
    if not matrix.ranks:
        raise ValueError("transport_matrix.ranks must not be empty")

    return ClusterProfile(
        profile_version=1,
        name=str(data.get("name", src.stem)),
        description=str(data.get("description", "")),
        runtime=runtime,
        scaling=scaling,
        stability=stability,
        transport_matrix=matrix,
        metadata=dict(data.get("metadata", {}) or {}),
        source_path=str(src),
    )


def apply_profile_env(env: dict[str, str], profile_env: dict[str, str]) -> dict[str, str]:
    out = dict(env)
    for k, v in profile_env.items():
        out[str(k)] = str(v)
    return out


def capture_env_snapshot(extra_keys: list[str] | None = None) -> dict[str, str]:
    keys = [
        "HOSTNAME",
        "SLURM_JOB_ID",
        "SLURM_CLUSTER_NAME",
        "SLURM_NNODES",
        "SLURM_NTASKS",
        "OMPI_COMM_WORLD_SIZE",
        "OMPI_COMM_WORLD_RANK",
        "MPICH_VERSION",
        "CUDA_VISIBLE_DEVICES",
        "NCCL_DEBUG",
        "OMP_NUM_THREADS",
    ]
    if extra_keys:
        for k in extra_keys:
            if str(k) and str(k) not in keys:
                keys.append(str(k))
    out: dict[str, str] = {}
    for k in keys:
        v = os.environ.get(k)
        if v is not None and str(v) != "":
            out[str(k)] = str(v)
    return out
