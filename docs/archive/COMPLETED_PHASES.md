# Completed Phases (archived from docs/TODO.md)

This file contains the completed task items that were removed from the active
TODO.md to keep it focused on planned/active work.

## Planned Maintenance - Many-Body TD Locality (completed)
- [x] PR-MB01 Many-body TD force-scope contract.
  - DoD: the runtime/observability distinction between full-system many-body evaluation and
    target-local many-body evaluation is explicit; `eam_td_breakdown_gpu` now records the frozen
    `pr_mb01_v1` baseline contract (`evaluation_scope`, `consumption_scope`,
    `target_local_available`, `target_local_force_calls`) for this track.
- [x] PR-MB02 CPU reference target-local `EAM/eam-alloy` TD path.
  - DoD: async CPU `td_local` many-body half-steps no longer rely on repeated full-system
    `forces_full(ctx.r)[ids0]`; they use `potential.forces_on_targets(...)` while preserving TD
    scheduling semantics. Strict materials gates and `longrun_envelope_ci` stay green.
- [x] PR-MB03 GPU refinement for target-local many-body TD path.
  - DoD: CUDA `EAM/eam-alloy` path consumes target/candidate-local work without repeated
    full-system many-body passes; `eam_td_breakdown_gpu` now reports current contract `pr_mb03_v1`
    against baseline `pr_mb01_v1` with `forces_full_share=0` on the representative `10K` benchmark.
  - DoD: CUDA `EAM/eam-alloy` path consumes target/candidate-local work without repeated
    full-system many-body passes; `eam_td_breakdown_gpu` shows reduced `forces_full` share on the
    representative `10K` benchmark.

## Planned Maintenance - TD Auto-Zoning (completed)
- [x] PR-ZA01 Resource-aware TD auto-zoning advisor.
  - DoD: detect available CPU/GPU/MPI resources, enumerate geometry-valid TD/space layouts,
    and emit a recommended zoning plan from benchmark evidence without changing runtime semantics.
  - Evidence sources should include `eam_decomp_zone_sweep_gpu` for layout search and
    `eam_td_breakdown_gpu` for separating true TD headroom from backend/runtime overhead.
  - Depends on: `PR-MB02` and `PR-MB03`, so zoning recommendations are not built on a false
    many-body ceiling.
  - Delivery note: `td_autozoning_advisor_gpu` is an operator-side VerifyLab preset and artifact
    workflow. It recommends a layout but does not auto-apply zoning policy at runtime.

## Planned Future Extension - ML Potentials (groundwork completed)
- [x] PR-ML01 ML-potential runtime contract + CPU reference harness.
  - DoD: versioned cutoff/descriptor/neighbor contract, CPU reference inference path, and explicit
    no-hidden-barrier semantics suitable for TD scheduling.
  - Delivery note: `potential.kind=ml/reference` now provides the contract-bearing CPU harness for
    serial / `td_local` / task-driven verification paths. Dedicated fixture-driven VerifyLab and
    interop acceptance remain the next step.
- [x] PR-ML02 ML-potential VerifyLab/interop groundwork.
  - DoD: strict fixture-driven CPU acceptance for at least one ML-potential family before any
    GPU refinement or auto-zoning policy is introduced for that track.
  - Delivery note: `ml_reference_smoke` now provides a strict task-based VerifyLab lane for the
    `quadratic_density` `ml/reference` family, and `scripts/ml_reference_parity_pack.py --strict`
    validates `examples/interop/ml_reference_suite_v1.json`.

## Phase A - Verification/Gating and Contract (completed)
- [x] PR-01 VerifyLab strict gating split (`smoke_ci` vs `smoke_regression`).
  - DoD: CI uses strict gate; non-strict runs remain diagnostic only.
- [x] PR-02 Align task schema with runtime behavior.
  - DoD: explicit validation for unsupported runtime fields; no silent schema/runtime mismatch.

## Phase B - CPU Metals/Alloys Readiness (completed)
- [x] PR-03 Remove single-type/single-mass runtime limitation for pair models.
  - DoD: binary/alloy tasks run in `serial` and `td_local`.
- [x] PR-04 Pair coefficient matrix by atom types (LJ/Morse).
  - DoD: type-dependent forces/energies verified against reference.
- [x] PR-05 Runtime support for `potential.kind=table` (not export-only).
  - DoD: table potential runs in `serial` and `td_local` with tests.
- [x] PR-06 CPU EAM/eam-alloy implementation (reference materials backend).
  - DoD: Al and alloy cases match LAMMPS reference within thresholds.

## Phase C - TD Integration for Materials (completed)
- [x] PR-07 Integrate EAM into `td_local` while preserving TD formal core.
  - DoD: verify parity vs CPU serial; no new invariant violations.
- [x] PR-08 Integrate EAM into `td_full_mpi` without new barriers.
  - DoD: MPI smoke/verify passes; `hG3/hV3/tG3` counters remain healthy.

## Phase D - Interop and Scientific Verification (completed)
- [x] PR-09 LAMMPS interop for metals/alloys (`eam/alloy`, elements/types mapping).
  - DoD: exported `data.lammps` + `in.lammps` run with minimal/no edits.
- [x] PR-10 VerifyLab presets for materials.
  - DoD: `metal_smoke` and `interop_metal_smoke` added with strict thresholds.
- [x] PR-11 Time-blocking/performance (`K>1`) formalization.
  - DoD: measured comm/compute improvement without correctness regressions.

## Phase E - GPU Backend (completed)
- [x] PR-G01 Backend abstraction (`cpu`/`cuda`, `run.device=auto|cpu|cuda`).
  - DoD: CPU behavior unchanged; backend-switch tests added.
- [x] PR-G02 GPU pair kernels (LJ/Morse) in serial/td_local.
  - DoD: GPU parity vs CPU within strict tolerances.
- [x] PR-G03 GPU neighbor/cell-list build path.
  - DoD: candidate sets/forces parity and baseline speedup report.
- [x] PR-G04 GPU table-potential runtime support.
  - DoD: `table` parity tests pass on CPU/GPU.
- [x] PR-G05 GPU EAM/eam-alloy kernels.
  - DoD: metals/alloys parity vs CPU/LAMMPS within thresholds.
- [x] PR-G06 TD-local pipeline on GPU (host automaton + device compute).
  - DoD: invariants preserved, no TD semantic drift.
- [x] PR-G07 Multi-GPU MPI (`1 rank = 1 GPU`, local-rank mapping).
  - DoD: 2/4 GPU smoke passes without global barrier insertion.
- [x] PR-G08 CUDA-aware MPI + overlap optimization (optional capability).
  - DoD: reduced comm overhead vs host-staging baseline.
- [x] PR-G09 GPU VerifyLab/CI presets.
  - DoD: `gpu_smoke`, `gpu_interop_smoke`, `gpu_metal_smoke` with `--strict`.
- [x] PR-G10 GPU profiling and optimization pack.
  - DoD: documented speedup, efficiency, and memory/perf budget in reports.

## Phase F - Ensemble Control (completed)
- [x] PR-E01 Ensemble input contract formalization (`nve/nvt/npt`) for task/config schemas.
  - DoD: strict schema validation rules documented and covered by tests.
- [x] PR-E02 Runtime guardrails for unsupported ensembles.
  - DoD: mode-specific guardrails enforced explicitly; no TD automaton/invariant changes.
- [x] PR-E03 CPU serial NVT thermostat implementation (Berendsen first).
  - DoD: stable temperature relaxation in serial path with deterministic tests.
- [x] PR-E04 CPU serial NPT barostat implementation (Berendsen first).
  - DoD: stable pressure/volume relaxation in serial path with deterministic tests.
- [x] PR-E05 TD-local NVT/NPT integration (same semantics contracts as serial).
  - DoD: `sync_mode=True` parity checks vs serial within strict thresholds.
- [x] PR-E06 TD-MPI NVT/NPT integration without new barriers.
  - DoD: `td_full_mpi` supports NVT generally; NPT guarded to `MPI size=1` to avoid cross-rank barostat barriers.
- [x] PR-E07 VerifyLab strict presets for ensembles (`nvt_smoke`, `npt_smoke`).
  - DoD: strict pass/fail presets and baseline artifacts for temperature/pressure envelopes.
- [x] PR-E08 LAMMPS interop parity for prepared NVT/NPT states.
  - DoD: documented import/export workflow and reproducible parity checks.

## Phase G - Universal Visualization and Analysis (completed)
- [x] PR-VZ01 Output schema governance (trajectory + metrics manifests, versioning).
  - DoD: contract documented in `docs/VISUALIZATION.md`; compatibility tests added.
- [x] PR-VZ02 Runtime output abstraction (mode-agnostic writer API).
  - DoD: identical mandatory trajectory columns across `serial`/`td_local`/`td_full_mpi`.
- [x] PR-VZ03 Extended trajectory channels (`xu/yu/zu`, `ix/iy/iz`, `fx/fy/fz`) behind explicit flags.
  - DoD: deterministic column matrix tests and backward compatibility validated.
- [x] PR-VZ04 Universal post-processing API (`scripts/viz_*`) with plugin hooks.
  - DoD: stable CSV/JSON outputs for displacement, crack geometry, species-mixing metrics.
- [x] PR-VZ05 External adapters (OVITO/VMD/ParaView) with reproducible presets.
  - DoD: one-command adapter entrypoints write artifacts under `results/<run_id>/viz/`.
- [x] PR-VZ06 Large-scale I/O policy (frame thinning, chunking, compression).
  - DoD: profiled throughput/storage tradeoff report and no strict-gate regressions.
- [x] PR-VZ07 Strict visualization verification gate (`viz_smoke`).
  - DoD: `scripts/run_verifylab_matrix.py --preset viz_smoke --strict` available and documented.
- [x] PR-VZ08 Documentation/prompt consolidation for visualization workflows.
  - DoD: `README.md`, `docs/VERIFYLAB.md`, `CODEX_MASTER_PROMPT.md`, and `AGENTS.md` aligned.

## Phase H - CUDA Execution Cycle (completed, CuPy RawKernel)
- [x] PR-C01 Governance refresh (docs/prompts only).
  - DoD: `AGENTS.md`, `docs/TODO.md`, `docs/CUDA_EXECUTION_PLAN.md`,
    `docs/MODE_CONTRACTS.md`, `CODEX_MASTER_PROMPT.md` are CUDA-cycle consistent.
- [x] PR-C02 GPU neighbor list kernel (CuPy RawKernel).
  - DoD: O(N) cell-list discovery on device, parity vs CPU `build_cell_list`, no TD semantic changes.
- [x] PR-C03 LJ/Morse pair force kernel (CuPy RawKernel).
  - DoD: single kernel launch per evaluation, O(N) memory, strict parity vs CPU.
- [x] PR-C04 Table + EAM/alloy fused kernels (CuPy RawKernel).
  - DoD: device-side interpolation and two-pass EAM without Python element loops; materials strict gates pass.
- [x] PR-C05 Persistent GPU state + transfer elimination.
  - DoD: positions/types stay on device across timesteps; longrun envelope and ensemble strict lanes green.
- [x] PR-C06 TD-MPI CUDA integration + CUDA stream overlap hardening.
  - DoD: overlap/cluster/transport strict lanes pass without new barriers.
- [x] PR-C07 Profiling + kernel optimization (Plan B decision point).
  - DoD: documented speedup vs Phase E baseline and CPU reference; evaluate C++/CUDA if needed.
- [x] PR-C08 Consolidation (docs/ops/playbook).
  - DoD: governance/docs fully synchronized; Phase E CuPy high-level code archived if replaced.
