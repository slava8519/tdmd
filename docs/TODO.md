## GPU Cycle (PR-0) — Backend Abstraction (v4.5.0)
- [ ] Add `tdmd/backends/base.py` ZoneComputeBackend.
- [ ] Add `tdmd/backends/cpu.py` CPUBackend.
- [ ] Wire td_local + td_full_mpi compute path through backend.
- [ ] Add backend equivalence tests.
- [ ] Keep `pytest -q` green.


# Task Board (Unified CPU + GPU, PR-sized)

> Orchestrator maintains this file.
> Strategy focus: TD-correct MD for metals and alloys, with CPU reference first and GPU as refinement.
> Baseline plan here is completed; post-baseline hardening (`v1`) is tracked in `docs/RISK_BURNDOWN_PLAN.md`; risk burndown `v2` is completed in `docs/RISK_BURNDOWN_PLAN_V2.md`.
> Active next cycle: GPU portability via Kokkos (`docs/PORTABILITY_KOKKOS_PLAN.md`, PR-K01..PR-K10).

## Mandatory Rules (all PRs)
- [ ] Preserve TD semantics: no bypass/merge of F/D/P/W/S, no implicit global barriers.
- [ ] Preserve invariants or add a new invariant + checks (pytest and/or VerifyLab metric/counter).
- [ ] GPU changes are refinement only: CPU path remains formal reference.
- [ ] Do not mark task complete if verification is non-strict only.
- [ ] Minimum quality gates for completion:
  - [ ] `.venv/bin/python -m pytest -q`
  - [ ] `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict`
  - [ ] `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --strict`
  - [ ] if PR touches ensemble behavior: `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset nvt_smoke --strict`
  - [ ] if PR touches ensemble behavior: `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset npt_smoke --strict`
  - [ ] if PR touches GPU: `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_smoke --strict`
  - [ ] if PR touches portability-track GPU backend policy: keep hardware-strict no-fallback rule active (`require_effective_gpu` / compatibility alias policy).
- [ ] Required docs updates when behavior changes:
  - [ ] `docs/SPEC_TD_AUTOMATON.md`
  - [ ] `docs/INVARIANTS.md`
  - [ ] `docs/THEORY_VS_CODE.md`
  - [ ] `docs/VERIFYLAB.md`
  - [ ] if PR touches visualization/output contract: `docs/VISUALIZATION.md`
  - [ ] if PR touches GPU: `docs/GPU_BACKEND.md` and `docs/VERIFYLAB_GPU.md`

## Critical Order (implementation sequence)
1. Stabilize verification gates and task/runtime contract (PR-01, PR-02).
2. Enable multi-component CPU pair model (PR-03, PR-04).
3. Add CPU materials physics reference (PR-05, PR-06).
4. Integrate materials physics into TD paths (PR-07, PR-08).
5. Expand interop + materials verification (PR-09, PR-10).
6. Optimize time-blocking/performance (PR-11).
7. GPU track only after CPU reference is valid: PR-G01 -> PR-G02 -> PR-G03 -> PR-G04 -> PR-G05 -> PR-G06 -> PR-G07 -> PR-G08 -> PR-G09 -> PR-G10.
8. GPU portability cycle (vendor-neutral backend via Kokkos): PR-K01 -> PR-K02 -> PR-K03 -> PR-K04 -> PR-K05 -> PR-K06 -> PR-K07 -> PR-K08 -> PR-K09 -> PR-K10.

## Phase A - Verification/Gating and Contract
- [x] PR-01 VerifyLab strict gating split (`smoke_ci` vs `smoke_regression`).
  - DoD: CI uses strict gate; non-strict runs remain diagnostic only.
- [x] PR-02 Align task schema with runtime behavior.
  - DoD: explicit validation for unsupported runtime fields; no silent schema/runtime mismatch.

## Phase B - CPU Metals/Alloys Readiness
- [x] PR-03 Remove single-type/single-mass runtime limitation for pair models.
  - DoD: binary/alloy tasks run in `serial` and `td_local`.
- [x] PR-04 Pair coefficient matrix by atom types (LJ/Morse).
  - DoD: type-dependent forces/energies verified against reference.
- [x] PR-05 Runtime support for `potential.kind=table` (not export-only).
  - DoD: table potential runs in `serial` and `td_local` with tests.
- [x] PR-06 CPU EAM/eam-alloy implementation (reference materials backend).
  - DoD: Al and alloy cases match LAMMPS reference within thresholds.

## Phase C - TD Integration for Materials
- [x] PR-07 Integrate EAM into `td_local` while preserving TD formal core.
  - DoD: verify parity vs CPU serial; no new invariant violations.
- [x] PR-08 Integrate EAM into `td_full_mpi` without new barriers.
  - DoD: MPI smoke/verify passes; `hG3/hV3/tG3` counters remain healthy.

## Phase D - Interop and Scientific Verification
- [x] PR-09 LAMMPS interop for metals/alloys (`eam/alloy`, elements/types mapping).
  - DoD: exported `data.lammps` + `in.lammps` run with minimal/no edits.
- [x] PR-10 VerifyLab presets for materials.
  - DoD: `metal_smoke` and `interop_metal_smoke` added with strict thresholds.
- [x] PR-11 Time-blocking/performance (`K>1`) formalization.
  - DoD: measured comm/compute improvement without correctness regressions.

## Phase E - GPU Backend (after Phase B baseline)
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

## Phase F - Ensemble Control (CPU reference first)
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

## Phase H - GPU Portability via Kokkos (active)
- [x] PR-K01 Portability governance + contract freeze (docs/prompts only).
  - DoD: `docs/PORTABILITY_KOKKOS_PLAN.md` created and all core prompts/governance docs aligned.
- [ ] PR-K02 Backend API expansion (`run.device=auto|cpu|cuda|hip|kokkos`) with safe fallback.
  - DoD: parser/CLI/tests updated; no TD semantic changes.
- [ ] PR-K03 VerifyLab hardware-strict generalization (`require_effective_gpu` + compatibility alias).
  - DoD: no-fallback policy is backend-agnostic and evidence artifacts updated.
- [ ] PR-K04 Kokkos runtime bridge + build integration.
  - DoD: reproducible build/runtime capability discovery without changing physics semantics.
- [ ] PR-K05 Kokkos LJ/Morse kernels for `serial`/`td_local`.
  - DoD: strict parity vs CPU within documented tolerances.
- [ ] PR-K06 Kokkos `table` + `eam/alloy` kernels.
  - DoD: materials strict gates + parity packs pass.
- [ ] PR-K07 td_local hardening on Kokkos path (sync/async).
  - DoD: longrun envelope and ensemble strict lanes remain green.
- [ ] PR-K08 td_full_mpi vendor-neutral multi-GPU mapping and transport hardening.
  - DoD: overlap/cluster/transport strict lanes pass without new barriers.
- [ ] PR-K09 Vendor lanes and cross-vendor strict parity governance.
  - DoD: NVIDIA+AMD hardware-strict lanes reject fallback and produce backend evidence.
- [ ] PR-K10 Consolidation (docs/prompts/ops handoff).
  - DoD: governance/docs/prompt set fully synchronized with implemented portability behavior.
