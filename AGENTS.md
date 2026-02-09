# AGENTS.md — Agent Roles for TDMD-TD

TDMD-TD implements a strict **Time Decomposition (TD)** molecular dynamics method based on an academic dissertation.
**Preserving theoretical correctness is mandatory.**

## Current Strategy
- Target domain: **metals and alloys**.
- Delivery order: CPU reference physics first, then TD integration, then GPU acceleration.
- GPU is optimization/refinement only; CPU remains the formal reference behavior.
- Baseline delivery (`docs/TODO.md`, PR-01..PR-11 and PR-G01..PR-G10) is completed.
- Post-baseline risk burndown hardening (`PR-H01..PR-H08`) is completed.
- Risk burndown `v2` (`docs/RISK_BURNDOWN_PLAN_V2.md`) is completed:
  - cluster performance/stability validation lanes are implemented,
  - expanded materials property-level references and strict gates are implemented.
- Next active cycle is **GPU portability (NVIDIA + AMD) via Kokkos**:
  - planning/governance source: `docs/PORTABILITY_KOKKOS_PLAN.md`,
  - execution order: `PR-K01..PR-K10`,
  - until cycle completion, runtime GPU implementation remains CUDA-first with strict no-fallback hardware gates.
- Current governance reference for mode guarantees and strict gates: `docs/MODE_CONTRACTS.md`.
- Universal visualization/analysis contract (`PR-VZ01..PR-VZ08`) is implemented.
  Ongoing work is maintenance/refinement under `docs/VISUALIZATION.md`.
- Versioning policy: Git tags/GitHub releases + `RELEASE_NOTES.md` (keep `README.md` current-state).

## Global Rules (apply to all agents)

1. **No semantic shortcuts**
   - Do NOT bypass TD states (F/D/P/W/S) or merge them for convenience.
   - Do NOT assume synchronous execution or global barriers unless explicitly modeled.
2. **Any new mechanism must**
   - preserve existing invariants, or
   - introduce a new invariant + verification (unit test or VerifyLab metric/counter).
3. **Optimizations are refinements only**
   - preserve formal-core behavior as reference semantics.
   - GPU path must stay numerically aligned with CPU path under defined tolerances.
4. **Every PR includes**
   - tests and strict verification runs,
   - documentation updates if behavior/contract changes.
5. **No false green**
   - non-strict diagnostic runs do not count as acceptance.
6. **No fallback green for hardware-strict tasks**
   - if task goal explicitly validates GPU backend/hardware path, CPU fallback is not considered success.

## Agent: Orchestrator
- Maintains `docs/ROADMAP.md`, `docs/TODO.md`, `docs/RISK_BURNDOWN_PLAN.md`, `docs/RISK_BURNDOWN_PLAN_V2.md`, `docs/PORTABILITY_KOKKOS_PLAN.md`, and `docs/VISUALIZATION.md`.
- Splits work into PR-sized tasks in critical order (verification -> CPU materials -> TD integration -> GPU).
- Enforces CI discipline, reproducibility, and strict gates.
- Assigns tasks to other agents.

## Agent: VerifyLab Engineer
Scope: scientific verification & regression.
- Owns: `tdmd/verify_v2.py`, `tdmd/verify_lab.py`, scripts under `scripts/`, tests under `tests/`.
- Must provide strict CI-grade presets:
  - `smoke_ci` (or equivalent strict smoke),
  - `interop_smoke`,
  - `metal_smoke`,
  - `interop_metal_smoke`,
  - `viz_smoke`,
  - `gpu_smoke`,
  - `gpu_interop_smoke`,
  - `gpu_metal_smoke`.
- Must deliver portability-cycle strict presets (per `docs/PORTABILITY_KOKKOS_PLAN.md`):
  - backend-agnostic hardware-strict policy (`require_effective_gpu`),
  - vendor lanes (`gpu_cuda_smoke_hw`, `gpu_hip_smoke_hw`),
  - cross-vendor parity lane (`gpu_portability_smoke`).
- Must maintain strict post-hardening gates:
  - `longrun_envelope_ci`,
  - materials parity pack (`scripts/materials_parity_pack.py --strict`),
  - strict-failure incident bundle generation/export path.
- Must maintain strict v2 lanes:
  - `cluster_scale_smoke`,
  - `cluster_stability_smoke`,
  - `mpi_transport_matrix_smoke`,
  - `metal_property_smoke`,
  - `interop_metal_property_smoke`.
- For hardware-strict GPU tasks, must provide evidence that requested GPU backend path was actually used (no CPU fallback acceptance).
- Must not change TD semantics or physics; only verification logic and thresholds policy.

## Agent: TD Automaton & Proof
Scope: formal semantics and correctness.
- Owns: `docs/SPEC_TD_AUTOMATON.md`, `docs/INVARIANTS.md`, invariant checks in code.
- Must map dissertation concepts to code modules.
- Must explicitly track ring-transfer core vs `static_rr` extension semantics.

## Agent: DepsGraph & 3D Scaling
Scope: dependency graphs and geometry.
- Owns: `DepsProvider*`, 3D AABB/PBC logic, general deps-graph (non-ring).
- Must preserve table-deps vs owner-deps separation.
- Must not introduce ring assumptions into static owner-routed modes.

## Agent: IO / Interop Engineer
Scope: task contract and external interoperability.
- Owns: `tdmd/io/*`, CLI I/O flags, interop fixtures under `examples/interop/`.
- Maintains versioned task schema with strict validation and explicit runtime-compatibility checks.
- Owns LAMMPS import/export compatibility for pair and materials workflows.

## Agent: Materials Potentials Engineer
Scope: physical models for metals/alloys.
- Owns: `tdmd/potentials.py`, force kernels, materials testcases.
- Must deliver multi-type/multi-mass runtime support and materials potentials (table, EAM/eam-alloy).
- Must preserve TD scheduling semantics while extending force-field capabilities.

## Agent: GPU Backend Engineer
Scope: accelerator implementation without semantic drift.
- Owns: backend abstraction, device kernels, GPU execution path, multi-GPU rank/device mapping.
- Must preserve CPU equivalence (within documented tolerances) and avoid new global barriers.
- Must provide profiling artifacts and deterministic testability guidance.
- Must expose/propagate fallback diagnostics so CI can distinguish real CUDA execution from CPU fallback.
- Must lead Kokkos portability implementation for NVIDIA+AMD while preserving CPU-reference semantics and current strict-gate policy.

## Agent: Visualization & Analysis Engineer
Scope: universal output contract, post-processing, rendering adapters.
- Owns: `tdmd/io/{trajectory,metrics}.py`, output schema manifests, and visualization scripts under `scripts/viz_*`.
- Must keep visualization as a passive observability layer (no feedback into integrator/automaton decisions).
- Must preserve mode/backend portability (`serial`/`td_local`/`td_full_mpi`, CPU/GPU).
- Must provide reproducible pipelines for external tools (OVITO/VMD/ParaView) via scripted presets.
- Must define versioned output schema and backward-compatibility policy.

## Required Documentation Updates (when relevant)
- `docs/SPEC_TD_AUTOMATON.md`
- `docs/INVARIANTS.md`
- `docs/THEORY_VS_CODE.md`
- `docs/VERIFYLAB.md`
- `docs/MODE_CONTRACTS.md` (if mode guarantees or strict-gate mapping changes)
- `docs/VISUALIZATION.md` (if output schema, adapters, or visualization gates change)
- GPU changes additionally require:
  - `docs/GPU_BACKEND.md`
  - `docs/VERIFYLAB_GPU.md`
- If risk status/ownership changes: update `docs/RISK_BURNDOWN_PLAN.md`, active cycle plan (`docs/RISK_BURNDOWN_PLAN_V2.md`), and portability cycle plan (`docs/PORTABILITY_KOKKOS_PLAN.md`).

## Acceptance Criteria
A task is complete only if:
- `pytest -q` passes,
- `scripts/run_verifylab_matrix.py --preset smoke_ci --strict` passes,
- `scripts/run_verifylab_matrix.py --preset interop_smoke --strict` passes,
- for long-horizon async verification/governance tasks:
  - `scripts/run_verifylab_matrix.py --preset longrun_envelope_ci --strict` passes,
- for ensemble-control tasks:
  - `scripts/run_verifylab_matrix.py --preset nvt_smoke --strict` passes,
  - `scripts/run_verifylab_matrix.py --preset npt_smoke --strict` passes,
- for materials/potential tasks:
  - `scripts/run_verifylab_matrix.py --preset metal_smoke --strict` passes,
  - `scripts/run_verifylab_matrix.py --preset interop_metal_smoke --strict` passes,
  - `scripts/materials_parity_pack.py --fixture examples/interop/materials_parity_suite_v1.json --config examples/td_1d_morse.yaml --strict` passes,
  - `scripts/materials_parity_pack.py --fixture examples/interop/materials_parity_suite_v2.json --config examples/td_1d_morse.yaml --strict` passes,
- for GPU-touching tasks:
  - `scripts/run_verifylab_matrix.py --preset gpu_smoke --strict` passes,
  - `scripts/run_verifylab_matrix.py --preset gpu_interop_smoke --strict` passes,
  - `scripts/run_verifylab_matrix.py --preset gpu_metal_smoke --strict` passes,
- for GPU portability-track tasks (`PR-K01+`):
  - all baseline GPU strict gates above still pass,
  - hardware-strict policy is backend-agnostic (`require_effective_gpu`),
  - CUDA-lane hardware-strict gate (`gpu_cuda_smoke_hw`) passes on NVIDIA runner,
  - HIP-lane hardware-strict gate (`gpu_hip_smoke_hw`) passes on AMD runner,
  - CPU fallback is treated as failure in hardware-strict vendor lanes,
- for MPI-overlap/cuda-aware tasks: `scripts/bench_mpi_overlap.py ...` artifact run passes and is attached,
- for hardware-strict GPU validation tasks: requested GPU backend execution is confirmed and CPU fallback is treated as failure,
- for strict-failure observability tasks: incident bundle is generated and contains manifest/checksums,
- for cluster validation tasks (v2):
  - `scripts/run_verifylab_matrix.py --preset cluster_scale_smoke --strict` passes,
  - `scripts/run_verifylab_matrix.py --preset cluster_stability_smoke --strict` passes,
  - `scripts/run_verifylab_matrix.py --preset mpi_transport_matrix_smoke --strict` passes,
- for materials property-reference tasks (v2):
  - `scripts/run_verifylab_matrix.py --preset metal_property_smoke --strict` passes,
  - `scripts/run_verifylab_matrix.py --preset interop_metal_property_smoke --strict` passes,
  - threshold policy artifact `golden/material_threshold_policy_v2.json` is present and parseable,
- for visualization/output-contract tasks:
  - contract/manifest tests pass,
  - `scripts/run_verifylab_matrix.py --preset viz_smoke --strict` passes,
  - produced artifacts include schema/version metadata,
- docs are updated appropriately,
- no invariant regressions (e.g., `hG3/hV3/tG3`) are introduced.



## Agent GPU Backend Engineer
Mission: implement GPU acceleration as a compute-backend refinement.

Constraints:
- MUST NOT change TD automaton semantics, deps rules, or liveness policy.
- MUST keep `W<=1` formal core intact.
- MUST keep verify v2 green.

Deliverables:
- Backend interface + CPUBackend.
- Feature-gated GPUBackend (default OFF).
- Minimal GPU correctness test.
- Verifylab integration for GPU benchmarks.
