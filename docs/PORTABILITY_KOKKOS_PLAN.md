# GPU Portability Plan (NVIDIA + AMD via Kokkos)

Status: planned, not started.  
Scope owner: Orchestrator + GPU Backend Engineer + VerifyLab Engineer.

## Goal
- Make GPU path portable across NVIDIA and AMD with Kokkos-based backend implementation.
- Preserve TD semantics (`F/D/P/W/S`) and invariants.
- Keep CPU path as formal reference behavior.
- Keep strict no-fallback acceptance policy for hardware-strict GPU validation.

## Non-Negotiable Constraints
- No semantic shortcut in TD automaton, dependency contracts, or ownership logic.
- No implicit global barriers.
- GPU remains refinement-only path.
- Any new backend mechanism must carry strict verification evidence and docs updates.

## Current Baseline (Before PR-K01)
- Runtime GPU path is CUDA-first.
- Strict gates already active: `smoke_ci`, `interop_smoke`, `gpu_smoke`, `gpu_interop_smoke`, `gpu_metal_smoke`, hardware-strict `*_hw`, cluster/material/property/viz lanes.
- This plan extends those gates; it does not replace them.

## Target Backend Contract (End-State)
- `run.device`: `auto|cpu|cuda|hip|kokkos`.
- Hardware-strict policy key: `require_effective_gpu` (with backward-compatible alias for `require_effective_cuda` during migration).
- Backend evidence fields in run artifacts:
  - `requested_device`,
  - `effective_device`,
  - `effective_gpu_api` (`cuda|hip|none`),
  - `fallback_from_gpu`,
  - `reason`,
  - `warnings`.
- TD-MPI transport flags remain semantics-preserving; naming may evolve to vendor-neutral aliases with compatibility shims.

## PR Plan (Critical Order)

### PR-K01: Governance + Contract Freeze (docs/prompts only)
- Scope:
  - Freeze portability requirements and PR order in docs/prompts.
  - Align strict-gate policy wording to vendor-neutral GPU semantics.
- DoD:
  - `AGENTS.md`, `docs/TODO.md`, `docs/ROADMAP.md`, `docs/MODE_CONTRACTS.md`, `CODEX_MASTER_PROMPT.md` are consistent.
  - No runtime/physics changes.
- Gates:
  - docs-only sanity: `pytest -q` (expected no behavior change).

### PR-K02: Backend API Expansion (control-plane, no kernel migration yet)
- Scope:
  - Expand backend selector and config validation to accept `hip` and `kokkos`.
  - Preserve current CUDA behavior and safe CPU fallback.
  - Keep existing `cuda` path unchanged.
- DoD:
  - Config/CLI parse new device values.
  - Backend evidence captures requested/effective backend consistently.
- Mandatory strict gates:
  - `pytest -q`
  - `scripts/run_verifylab_matrix.py --preset smoke_ci --strict`
  - `scripts/run_verifylab_matrix.py --preset interop_smoke --strict`
  - `scripts/run_verifylab_matrix.py --preset gpu_smoke --strict`

### PR-K03: VerifyLab Hardware-Strict Generalization
- Scope:
  - Introduce backend-agnostic strict key `require_effective_gpu`.
  - Keep compatibility with legacy `require_effective_cuda`.
  - Add vendor-specific hardware presets (CUDA/HIP) in matrix policy.
- DoD:
  - Hardware-strict failures are based on requested GPU backend not resolving effectively.
  - Artifact schema updated and documented.
- Mandatory strict gates:
  - `pytest -q`
  - `scripts/run_verifylab_matrix.py --preset smoke_ci --strict`
  - `scripts/run_verifylab_matrix.py --preset interop_smoke --strict`
  - `scripts/run_verifylab_matrix.py --preset gpu_smoke --strict`
  - `scripts/run_verifylab_matrix.py --preset gpu_interop_smoke --strict`
  - `scripts/run_verifylab_matrix.py --preset gpu_metal_smoke --strict`

### PR-K04: Kokkos Runtime Bridge + Build Integration
- Scope:
  - Add C++ bridge module with Kokkos init/finalize/runtime query hooks.
  - Keep force computation behavior unchanged until kernel PRs.
- DoD:
  - Build flow documented and reproducible.
  - Runtime can report backend capability without changing TD semantics.
- Mandatory strict gates:
  - all PR-K03 gates.

### PR-K05: Kokkos LJ/Morse Pair Kernels
- Scope:
  - Implement pair kernels in Kokkos backend.
  - Keep candidate-set semantics equivalent to CPU path.
- DoD:
  - Parity tests pass for `serial` and `td_local`.
  - No invariant regressions.
- Mandatory strict gates:
  - all PR-K03 gates,
  - `scripts/run_verifylab_matrix.py --preset longrun_envelope_ci --strict` (async drift regression guard).

### PR-K06: Kokkos Table + EAM/eam-alloy Kernels
- Scope:
  - Port `table` and `eam/alloy` GPU execution to Kokkos backend.
  - Preserve current force/observable parity expectations.
- DoD:
  - Materials strict suites pass with Kokkos-backed GPU path.
  - No change in TD scheduling semantics.
- Mandatory strict gates:
  - all PR-K05 gates,
  - `scripts/run_verifylab_matrix.py --preset metal_smoke --strict`
  - `scripts/run_verifylab_matrix.py --preset interop_metal_smoke --strict`
  - `scripts/materials_parity_pack.py --fixture examples/interop/materials_parity_suite_v1.json --config examples/td_1d_morse.yaml --strict`
  - `scripts/materials_parity_pack.py --fixture examples/interop/materials_parity_suite_v2.json --config examples/td_1d_morse.yaml --strict`

### PR-K07: TD-local Integration Hardening (Kokkos path)
- Scope:
  - Wire td_local host automaton + Kokkos compute path.
  - Validate sync/async behavior remains contract-compliant.
- DoD:
  - Sync parity and async envelope remain within strict thresholds.
- Mandatory strict gates:
  - all PR-K06 gates,
  - `scripts/run_verifylab_matrix.py --preset nvt_smoke --strict`
  - `scripts/run_verifylab_matrix.py --preset npt_smoke --strict`

### PR-K08: TD-MPI + Multi-GPU Vendor-Neutral Transport
- Scope:
  - Generalize rank/device mapping for CUDA/HIP backends.
  - Keep overlap/transport semantics as refinement-only behavior.
- DoD:
  - MPI overlap and transport matrix lanes remain strict-green.
  - No new barriers introduced.
- Mandatory strict gates:
  - all PR-K07 gates,
  - `scripts/run_verifylab_matrix.py --preset mpi_overlap_smoke --strict`
  - `scripts/run_verifylab_matrix.py --preset mpi_overlap_cudaaware_smoke --strict`
  - `scripts/run_verifylab_matrix.py --preset cluster_scale_smoke --strict`
  - `scripts/run_verifylab_matrix.py --preset cluster_stability_smoke --strict`
  - `scripts/run_verifylab_matrix.py --preset mpi_transport_matrix_smoke --strict`

### PR-K09: Vendor Lanes + Cross-Vendor Parity Governance
- Scope:
  - Activate strict vendor lanes in CI/cluster policy:
    - NVIDIA hardware-strict lane,
    - AMD hardware-strict lane.
  - Add cross-vendor parity reporting policy.
- DoD:
  - Hardware-strict GPU lanes reject CPU fallback on both vendors.
  - Artifacts include vendor/backend evidence.
- Mandatory strict gates:
  - all PR-K08 gates,
  - property-level materials strict lanes:
    - `scripts/run_verifylab_matrix.py --preset metal_property_smoke --strict`
    - `scripts/run_verifylab_matrix.py --preset interop_metal_property_smoke --strict`

### PR-K10: Final Consolidation (docs/prompts/ops handoff)
- Scope:
  - Finalize docs, prompts, and operator playbooks for portable GPU cycle.
  - Keep backward-compatible CLI/task guidance where possible.
- DoD:
  - Governance docs internally consistent.
  - Execution playbook explicitly maps each strict gate to ownership.
- Mandatory strict gates:
  - full strict matrix from PR-K09,
  - `scripts/run_verifylab_matrix.py --preset viz_smoke --strict`.

## Planned New/Updated Presets
- New (planned): `gpu_cuda_smoke_hw`, `gpu_hip_smoke_hw`, `gpu_portability_smoke`.
- Compatibility policy:
  - existing presets remain valid during migration,
  - aliases may map old names to new policy keys until final consolidation.

## Start Condition for Implementation
- Implementation begins only after this plan and all referenced prompts/docs are aligned.
- First implementation PR must be `PR-K02`.
