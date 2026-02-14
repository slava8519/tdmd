# CUDA Execution Plan (Current Cycle)

Status: active.
Scope owner: Orchestrator + GPU Backend Engineer + VerifyLab Engineer.

## Goal
- Continue GPU acceleration on NVIDIA CUDA only.
- Preserve TD semantics (`F/D/P/W/S`) and invariants.
- Keep CPU path as formal reference behavior.
- Keep strict no-fallback acceptance policy for hardware-strict CUDA validation.

## Technical Direction
- Primary stack for near-term PRs: `numba-cuda`.
- Plan B for performance/production hardening (after contract + VerifyLab v2 stabilization):
  - `CuPy RawKernel`, or
  - C++/CUDA extension.

## Non-Negotiable Constraints
- No semantic shortcut in TD automaton, dependency contracts, or ownership logic.
- No implicit global barriers.
- GPU remains refinement-only path.
- Any new mechanism must carry strict verification evidence and docs updates.

## Current Baseline
- Runtime GPU path is CUDA-first.
- Strict gates already active: `smoke_ci`, `interop_smoke`, `gpu_smoke`,
  `gpu_interop_smoke`, `gpu_metal_smoke`, `gpu_smoke_hw`,
  cluster/material/property/viz lanes.

## PR Plan (Critical Order)

### PR-C01: CUDA Governance Refresh (docs/prompts only)
- Scope:
  - Replace old portability/Kokkos planning with CUDA-only governance.
  - Align prompts/contracts/TODO to current execution strategy.
- DoD:
  - `AGENTS.md`, `docs/TODO.md`, `docs/ROADMAP_GPU.md`, `docs/PR_PLAN_GPU.md`,
    `docs/MODE_CONTRACTS.md`, `CODEX_MASTER_PROMPT.md` are consistent.
- Gates:
  - docs-only sanity: `.venv/bin/python -m pytest -q`.

### PR-C02: Numba-CUDA Backend Scaffold
- Scope:
  - Add Numba-CUDA backend entry path and capability detection.
  - Keep current CuPy CUDA path intact as reference fallback.
  - No force-kernel migration yet.
- DoD:
  - Backend evidence records requested/effective CUDA backend path.
  - No TD semantic changes.
- Mandatory strict gates:
  - `.venv/bin/python -m pytest -q`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_smoke_hw --strict`

### PR-C03: Numba-CUDA LJ/Morse Kernels (`serial`/`td_local`)
- Scope:
  - Port pair kernels with parity thresholds aligned to CPU reference.
- DoD:
  - No invariant regressions.
  - Candidate-set semantics unchanged.
- Mandatory strict gates:
  - all PR-C02 gates,
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_interop_smoke --strict`

### PR-C04: Numba-CUDA Table + EAM/eam-alloy Kernels
- Scope:
  - Port table and materials kernels on CUDA.
- DoD:
  - Materials parity and strict gates remain green.
- Mandatory strict gates:
  - all PR-C03 gates,
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_metal_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset metal_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_metal_smoke --strict`

### PR-C05: TD-Local Hardening + Longrun Envelope
- Scope:
  - Harden sync/async behavior on CUDA path without semantic drift.
- DoD:
  - Long-horizon envelope and ensemble gates remain strict-green.
- Mandatory strict gates:
  - all PR-C04 gates,
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset longrun_envelope_ci --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset nvt_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset npt_smoke --strict`

### PR-C06: TD-MPI CUDA Mapping + Overlap Hardening
- Scope:
  - Harden rank/device mapping and overlap transport on CUDA-only path.
- DoD:
  - MPI overlap/cluster/transport strict lanes remain green.
- Mandatory strict gates:
  - all PR-C05 gates,
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset mpi_overlap_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset mpi_overlap_cudaaware_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset cluster_scale_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset cluster_stability_smoke --strict`
  - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset mpi_transport_matrix_smoke --strict`

### PR-C07: Plan B Performance Track (post-stabilization)
- Scope:
  - Evaluate and optionally implement one of:
    - `CuPy RawKernel`,
    - C++/CUDA extension.
- Guardrail:
  - Start only after PR-C06 strict matrix is stable.
- DoD:
  - Demonstrated perf gain without VerifyLab regressions.

### PR-C08: Consolidation (docs/ops/playbook)
- Scope:
  - Finalize CUDA-only governance, runbooks, and CI/operator guidance.
- DoD:
  - Documentation and strict-gate ownership are fully synchronized.

## Start Condition
- Implementation starts from PR-C02.
- PR-C01 is docs-only alignment and can be shipped immediately.
