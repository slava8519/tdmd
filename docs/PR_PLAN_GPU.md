# GPU PR Plan (CUDA-Only)

This file is the compact PR queue for GPU work.
Source of truth (full details): `docs/CUDA_EXECUTION_PLAN.md`.

## Current Direction
- Backend target: NVIDIA CUDA only.
- Primary implementation stack: `numba-cuda`.
- Plan B after stabilization: `CuPy RawKernel` or C++/CUDA extension.

## Active PR Queue
- PR-C01: CUDA governance refresh (docs/prompts only).
- PR-C02: Numba-CUDA backend scaffold (no kernel migration yet).
- PR-C03: Numba-CUDA LJ/Morse kernels (`serial`/`td_local`).
- PR-C04: Numba-CUDA table + EAM/eam-alloy kernels.
- PR-C05: TD-local hardening + longrun envelope.
- PR-C06: TD-MPI CUDA mapping + overlap hardening.
- PR-C07: Plan B performance track (RawKernel or C++/CUDA extension).
- PR-C08: consolidation (docs/ops/playbook).

## Invariants
- Preserve TD semantics (`F/D/P/W/S`) and formal-core invariant `W<=1`.
- CPU remains formal reference behavior.
- Hardware-strict CUDA validation treats CPU fallback as failure.
