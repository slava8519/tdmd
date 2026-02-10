from __future__ import annotations

import numpy as np

from .backend import ComputeBackend
from .forces_gpu import (
    forces_on_targets_celllist_backend,
    forces_on_targets_pair_backend,
    supports_pair_gpu,
)


def try_gpu_forces_on_targets(
    *,
    r: np.ndarray,
    box: float,
    cutoff: float,
    rc: float,
    potential,
    target_ids: np.ndarray,
    candidate_ids: np.ndarray,
    atom_types: np.ndarray,
    backend: ComputeBackend,
) -> np.ndarray | None:
    """Try the GPU force path (CUDA) and return forces or None if unavailable.

    Contract: This function is a refinement only; callers must preserve CPU-reference
    semantics by providing a CPU fallback if None is returned.
    """
    if backend.device != "cuda":
        return None
    if not supports_pair_gpu(potential):
        return None

    f_gpu = forces_on_targets_celllist_backend(
        r=r,
        box=float(box),
        cutoff=float(cutoff),
        rc=float(rc),
        potential=potential,
        target_ids=target_ids,
        candidate_ids=candidate_ids,
        atom_types=atom_types,
        backend=backend,
    )
    if f_gpu is not None:
        return np.asarray(f_gpu, dtype=float)

    f_pair = forces_on_targets_pair_backend(
        r=r,
        box=float(box),
        cutoff=float(cutoff),
        potential=potential,
        target_ids=target_ids,
        candidate_ids=candidate_ids,
        atom_types=atom_types,
        backend=backend,
    )
    if f_pair is not None:
        return np.asarray(f_pair, dtype=float)

    return None
