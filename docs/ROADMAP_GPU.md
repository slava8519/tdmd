# GPU Roadmap (CUDA-Only)

Principle:
- GPU is compute-backend refinement only.
- TD automaton/control semantics stay on CPU reference path.

## Completed Baseline (Phase E)
- CUDA-first runtime path via CuPy high-level API.
- Strict GPU presets exist (`gpu_smoke`, `gpu_interop_smoke`, `gpu_metal_smoke`, `gpu_smoke_hw`).
- Functionally correct but performance-limited (O(N²) memory, Python-level loops, no persistent GPU state).

## Active Cycle (Phase H)
- CUDA execution cycle: `PR-C01 -> PR-C08` is complete.
- Full plan: `docs/CUDA_EXECUTION_PLAN.md`.
- Compact queue: `docs/PR_PLAN_GPU.md`.

## Post-CUDA Maintenance Order
1. **Many-body TD locality contract** for `EAM/eam-alloy` so the runtime distinguishes
   full-system many-body evaluation from target-local TD evaluation explicitly.
2. **CPU-reference target-local many-body path** to remove the current `forces_full` ceiling
   before tuning heuristics.
3. **GPU refinement of the corrected many-body path** rather than optimizing around the old
   full-system ceiling.
4. **Resource-aware TD auto-zoning advisor** only after the many-body locality model is reliable.
5. **ML-potential extension** on top of the same many-body locality contract, CPU reference first.

## Stack Policy
- Primary stack: **CuPy RawKernel** (CUDA C kernels compiled via CuPy).
- Plan B (if RawKernel ceiling reached): C++/CUDA extension module.
- Numba-CUDA is not in scope (CuPy already covers the dependency; RawKernel gives
  full CUDA C control without adding a JIT layer).

## Key Improvements Targeted
1. **O(N) neighbor lists** replacing O(N²) broadcast matrices.
2. **Single kernel launch** per force evaluation replacing Python for-loops.
3. **Fused EAM kernels** replacing Python element-type iteration.
4. **Persistent GPU state** eliminating per-step host↔device transfers.
5. **CUDA stream overlap** for MPI halo exchange + force computation.

## Current State
- Current production decision: stay on `CuPy RawKernel`.
- Representative profiler workflow:
  `python scripts/profile_gpu_backend.py --config examples/td_1d_morse.yaml --out-csv results/gpu_profile.csv --out-md results/gpu_profile.md --out-json results/gpu_profile.summary.json --require-effective-cuda --eam-n-atoms 512 --eam-steps 2`
- Optional `Phase E` comparison uses commit `efb864e` via a temporary git worktree.
- Large-run `EAM/eam-alloy` observability now also includes:
  - `eam_decomp_zone_sweep_gpu` for layout search,
  - `eam_td_breakdown_gpu` for attributing runtime before changing TD zoning policy.

## Acceptance Baseline (always)
- `.venv/bin/python -m pytest -q`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --strict`
- GPU-touching work additionally requires strict GPU presets per `docs/VERIFYLAB.md`.
