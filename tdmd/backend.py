from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ComputeBackend:
    device: str
    xp: object
    cuda_available: bool
    reason: str = ""


def _detect_cupy() -> tuple[object | None, str]:
    try:
        import cupy as cp  # type: ignore
    except Exception as exc:
        return None, f"cupy import failed: {exc}"
    try:
        ndev = int(cp.cuda.runtime.getDeviceCount())
        if ndev <= 0:
            return None, "CUDA runtime reports 0 devices"
    except Exception as exc:
        return None, f"CUDA runtime unavailable: {exc}"
    return cp, ""


def resolve_backend(device: str = "auto") -> ComputeBackend:
    req = str(device or "auto").strip().lower()
    if req not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be one of: auto, cpu, cuda")

    cp, cp_reason = _detect_cupy()
    cuda_ok = cp is not None

    if req == "cpu":
        return ComputeBackend(device="cpu", xp=np, cuda_available=cuda_ok, reason="forced cpu")

    if req == "cuda":
        if cuda_ok:
            return ComputeBackend(device="cuda", xp=cp, cuda_available=True, reason="")
        warnings.warn(
            f"CUDA backend requested but unavailable ({cp_reason}); falling back to CPU",
            RuntimeWarning,
        )
        return ComputeBackend(device="cpu", xp=np, cuda_available=False, reason=cp_reason)

    # auto
    force_cpu = os.environ.get("TDMD_FORCE_CPU", "").strip().lower() in ("1", "true", "yes", "on")
    if force_cpu:
        return ComputeBackend(device="cpu", xp=np, cuda_available=cuda_ok, reason="TDMD_FORCE_CPU")
    if cuda_ok:
        return ComputeBackend(device="cuda", xp=cp, cuda_available=True, reason="auto")
    return ComputeBackend(device="cpu", xp=np, cuda_available=False, reason=cp_reason)


def local_rank_from_env(env: dict[str, str] | None = None) -> int:
    e = os.environ if env is None else env
    for key in ("OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "SLURM_LOCALID"):
        raw = str(e.get(key, "")).strip()
        if raw:
            try:
                return max(0, int(raw))
            except Exception:
                continue
    return 0


def cuda_device_for_local_rank(local_rank: int, device_count: int) -> int:
    ndev = int(device_count)
    if ndev <= 0:
        raise ValueError("device_count must be positive")
    return int(max(0, int(local_rank))) % ndev


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        import cupy as cp  # type: ignore
    except Exception:
        return np.asarray(x)
    if isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)
