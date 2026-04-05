# GPU PR Plan (Post-CUDA Maintenance)

This file is the compact PR queue for GPU work.
Source of truth (full details): `docs/CUDA_EXECUTION_PLAN.md`.
Current local dirty-worktree decomposition: `docs/WORKTREE_SLICE_PLAN.md`.

## Current Direction
- Backend target: NVIDIA CUDA only.
- Primary implementation stack: **CuPy RawKernel** (CUDA C via CuPy).
- Plan B (if performance ceiling reached): C++/CUDA extension module.

## Completed CUDA Cycle
- PR-C01: CUDA governance refresh (docs/prompts only). **COMPLETE**
- PR-C02: GPU neighbor list kernel (CuPy RawKernel, O(N) cell-list on device). **COMPLETE**
- PR-C03: LJ/Morse pair force kernel (CuPy RawKernel, single launch per evaluation). **COMPLETE**
- PR-C04: Table + EAM/alloy fused kernels (CuPy RawKernel, no Python element loops). **COMPLETE**
- PR-C05: Persistent GPU state + host↔device transfer elimination. **COMPLETE**
- PR-C06: TD-MPI CUDA integration + CUDA stream overlap hardening. **COMPLETE**
- PR-C07: Profiling + kernel optimization (Plan B decision point). **COMPLETE**
- PR-C08: Consolidation (docs/ops/playbook). **COMPLETE**

## Active Maintenance Queue
- PR-MB01: many-body TD force-scope contract + baseline observability. **NEXT**
  - Goal: make the distinction between full-system many-body evaluation and target-local
    evaluation explicit for TD runtimes.
  - Evidence: freeze `eam_td_breakdown_gpu` as the representative large-run artifact.
- PR-MB02: CPU-reference target-local `EAM/eam-alloy` TD path.
  - Goal: remove repeated full-system `forces_full(ctx.r)[ids0]` dependence from async many-body
    `td_local` half-steps without changing TD semantics.
- PR-MB03: GPU target-local many-body refinement for `EAM/eam-alloy`.
  - Goal: make CUDA consume the corrected target/candidate-local many-body path rather than the
    current full-system ceiling.
- PR-ZA01: resource-aware TD auto-zoning advisor.
  - Goal: detect resources, enumerate strict-valid layouts, and emit recommendations only after
    PR-MB02/PR-MB03 make the many-body cost model trustworthy.

## Future ML-Potential Dependency
- Any ML-potential acceleration track should start only after PR-MB01..PR-MB03 define the
  many-body locality contract clearly.
- ML potentials should arrive CPU-reference first, then strict VerifyLab/interop coverage, then GPU
  refinement and zoning policy.

## Current Decision
- Stay on `CuPy RawKernel` as the active production stack.
- Re-open Plan B only if representative `gpu-profile` evidence shows RawKernel no longer
  beating both CPU reference and the archived `Phase E` baseline.

## Invariants
- Preserve TD semantics (`F/D/P/W/S`) and formal-core invariant `W<=1`.
- CPU remains formal reference behavior.
- Hardware-strict CUDA validation treats CPU fallback as failure.
