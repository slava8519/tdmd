# CUDA Execution Plan (Current Cycle)

Status: active.
Scope owner: Orchestrator + GPU Backend Engineer + VerifyLab Engineer.

## Goal
- Replace proof-of-concept CuPy high-level GPU path with production-quality CUDA kernels.
- Preserve TD semantics (`F/D/P/W/S`) and invariants.
- Keep CPU path as formal reference behavior.
- Keep strict no-fallback acceptance policy for hardware-strict CUDA validation.

## Technical Direction
- Primary stack: **CuPy RawKernel** (CUDA C kernels compiled and launched via CuPy).
- Plan B (if RawKernel ceiling reached): C++/CUDA extension module.
- Numba-CUDA is **not** in scope — CuPy is already a dependency, RawKernel provides
  full CUDA C control without adding another runtime.

## Rationale for CuPy RawKernel over Numba-CUDA
- CuPy is already in the dependency stack (Phase E baseline).
- RawKernel exposes real CUDA C — full control over thread/block layout, shared memory,
  coalesced memory access, warp-level primitives.
- CUDA C code from RawKernel is trivially portable to a `.cu` file if C++/CUDA extension
  is needed later.
- No new dependency or JIT compiler; CuPy handles compilation and caching.
- Numba-CUDA would add a dependency and a separate JIT layer with less kernel-level
  control than raw CUDA C.

## Current Baseline (what exists)
The Phase E implementation (`tdmd/forces_gpu.py`) is functional but has critical
performance limitations:
- **O(N²) memory**: pair forces use full broadcast `(N_t × N_c × 3)` distance matrices.
- **Python-level for-loops**: cell-list path dispatches one GPU call per atom.
- **EAM nested loops**: triple Python loop over element types for many-body forces.
- **No persistent GPU state**: every call transfers positions host→device and forces
  device→host.

These are the problems the current cycle solves.

## Non-Negotiable Constraints
- No semantic shortcut in TD automaton, dependency contracts, or ownership logic.
- No implicit global barriers.
- GPU remains refinement-only path.
- Any new mechanism must carry strict verification evidence and docs updates.

## PR Plan (Critical Order)

### PR-C01: CUDA Governance Refresh (docs/prompts only) — COMPLETE
- Scope:
  - Replace old portability/Kokkos planning with CUDA-only governance.
  - Align prompts/contracts/TODO to current execution strategy.
- DoD:
  - `AGENTS.md`, `docs/TODO.md`, `docs/ROADMAP_GPU.md`, `docs/PR_PLAN_GPU.md`,
    `docs/MODE_CONTRACTS.md`, `CODEX_MASTER_PROMPT.md` are consistent.
- Gates:
  - docs-only sanity: `.venv/bin/python -m pytest -q`.

### PR-C02: GPU Neighbor List Kernel (CuPy RawKernel)
- Scope:
  - Implement cell-list neighbor-pair discovery as a single CUDA kernel.
  - Replace Python for-loop cell-list path (`forces_gpu.py:219-243`) with one kernel launch.
  - Output: compact neighbor list (pair indices + distances) on device.
  - Memory: O(N × max_neighbors) instead of O(N²).
- DoD:
  - Neighbor list parity vs CPU `build_cell_list` within tolerances.
  - No TD semantic changes. Backend abstraction (`backend.py`) unchanged.
- Mandatory strict gates:
  - `.venv/bin/python -m pytest -q`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_smoke_hw --strict`

### PR-C03: LJ/Morse Pair Force Kernel (CuPy RawKernel)
- Scope:
  - CUDA C kernel: one thread per atom, iterate neighbor list, accumulate forces.
  - Replace CuPy broadcast pair computation (`forces_gpu.py:95-118`).
  - Support per-type pair coefficients via device-side lookup tables.
- DoD:
  - Force parity vs CPU within documented tolerances.
  - O(N) memory instead of O(N²).
  - Single kernel launch for all target atoms.
- Mandatory strict gates:
  - all PR-C02 gates,
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_interop_smoke --strict`

### PR-C04: Table + EAM/alloy Kernels (CuPy RawKernel)
- Scope:
  - Table potential: CUDA kernel with device-side interpolation (texture or global memory).
  - EAM/alloy: two-pass CUDA kernel (density accumulation → embedding derivative → force).
  - Replace Python element-type loops (`forces_gpu.py:153-171`).
- DoD:
  - Materials parity and strict gates pass.
  - EAM kernel handles multi-element without Python-level loops.
- Mandatory strict gates:
  - all PR-C03 gates,
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_metal_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset metal_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_metal_smoke --strict`

### PR-C05: Persistent GPU State + Transfer Elimination
- Scope:
  - Keep positions/types on device across timesteps (avoid per-step host→device copy).
  - Only transfer updated halo regions (delta updates).
  - Device-resident force output (avoid per-step device→host copy where possible).
  - Integrate with `serial` and `td_local` execution paths.
- DoD:
  - Measurable reduction in H2D/D2H bytes per step.
  - Long-run envelope and ensemble gates remain strict-green.
- Mandatory strict gates:
  - all PR-C04 gates,
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset longrun_envelope_ci --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset nvt_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset npt_smoke --strict`

### PR-C06: TD-MPI CUDA Integration + Overlap Hardening
- Scope:
  - Integrate RawKernel force path into `td_full_mpi`.
  - Overlap force computation (CUDA stream) with MPI halo exchange.
  - Rank→device mapping preserved from Phase E.
- DoD:
  - MPI overlap/cluster/transport strict lanes pass without new barriers.
- Mandatory strict gates:
  - all PR-C05 gates,
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset mpi_overlap_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset mpi_overlap_cudaaware_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset cluster_scale_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset cluster_stability_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset mpi_transport_matrix_smoke --strict`

### PR-C07: Profiling + Kernel Optimization
- Scope:
  - Profile kernel occupancy, memory throughput, launch overhead.
  - Optimize: shared memory for neighbor data, coalesced access patterns, warp-level
    reductions for EAM density pass.
  - Evaluate whether C++/CUDA extension is needed (Plan B decision point).
- Guardrail:
  - Start only after PR-C06 strict matrix is stable.
- DoD:
  - Documented speedup vs Phase E baseline and vs CPU reference.
  - No VerifyLab regressions.

### PR-C08: Consolidation (docs/ops/playbook)
- Scope:
  - Finalize CUDA governance, runbooks, and CI/operator guidance.
  - Archive Phase E CuPy high-level code if fully replaced.
- DoD:
  - Documentation and strict-gate ownership are fully synchronized.

## Start Condition
- Implementation starts from PR-C02.
- PR-C01 is docs-only alignment and is complete.
