# GPU PR Plan (CUDA-Only)

This file is the compact PR queue for GPU work.
Source of truth (full details): `docs/CUDA_EXECUTION_PLAN.md`.

## Current Direction
- Backend target: NVIDIA CUDA only.
- Primary implementation stack: **CuPy RawKernel** (CUDA C via CuPy).
- Plan B (if performance ceiling reached): C++/CUDA extension module.

## Active PR Queue
- PR-C01: CUDA governance refresh (docs/prompts only). **COMPLETE**
- PR-C02: GPU neighbor list kernel (CuPy RawKernel, O(N) cell-list on device).
- PR-C03: LJ/Morse pair force kernel (CuPy RawKernel, single launch per evaluation).
- PR-C04: Table + EAM/alloy fused kernels (CuPy RawKernel, no Python element loops).
- PR-C05: Persistent GPU state + host↔device transfer elimination.
- PR-C06: TD-MPI CUDA integration + CUDA stream overlap hardening.
- PR-C07: Profiling + kernel optimization (Plan B decision point).
- PR-C08: Consolidation (docs/ops/playbook).

## Invariants
- Preserve TD semantics (`F/D/P/W/S`) and formal-core invariant `W<=1`.
- CPU remains formal reference behavior.
- Hardware-strict CUDA validation treats CPU fallback as failure.
