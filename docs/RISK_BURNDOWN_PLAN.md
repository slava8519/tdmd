# Risk Burndown Plan (Post-Baseline)

This plan starts after completion of `docs/TODO.md` (PR-01..PR-11, PR-G01..PR-G10).
Status: completed and archived. Follow-up cycle `docs/RISK_BURNDOWN_PLAN_V2.md` is also completed.
Current active follow-up cycle for GPU portability is tracked in `docs/PORTABILITY_KOKKOS_PLAN.md`.

## Goal
Reduce technical and scientific risks while preserving TD formal semantics for metals/alloys workflows.

## Current Status
- `PR-H01..PR-H08` are completed.
- Governance reference for mode guarantees and strict gates: `docs/MODE_CONTRACTS.md`.

## Non-Negotiable Rules
- Keep TD states and transitions unchanged (`F/D/P/W/S`), unless explicitly specified in formal docs and verified.
- Do not add implicit global barriers.
- CPU path remains formal reference; GPU is refinement only.
- Every behavior change requires tests + strict VerifyLab + docs updates.

## Current Risk Register
| ID | Risk | Severity | Likelihood | Why it matters |
|---|---|---:|---:|---|
| R1 | GPU strict passes can hide CPU fallback | High | High | false confidence in CUDA path correctness/perf |
| R2 | MPI overlap/cuda-aware path under-validated on real clusters | High | Medium | latent deadlocks/regressions at scale |
| R3 | Materials validation is not deep enough at property level | High | Medium | scientifically incorrect conclusions for alloys |
| R4 | Warning-only geometry/config guardrails in strict pipelines | Medium | Medium | bad configurations may slip through |
| R5 | MPI uniform-mass limitation in task path | Medium | Medium | reduces alloy realism and interop value |
| R6 | Long-horizon stability envelopes are underspecified | Medium | Medium | drift regressions are hard to catch early |
| R7 | Failure triage observability is fragmented | Medium | High | slow debugging and weak reproducibility |

## Critical Order (PR-sized)

### [x] PR-H01: Hardware-strict GPU lane (no-fallback acceptance)
- Risk target: R1
- Scope:
  - Add a strict gate that fails if `run.device=cuda` resolves to CPU fallback.
  - Persist backend evidence (`device`, fallback flag, reason) in run artifacts.
- DoD:
  - Dedicated CUDA runner executes `gpu_smoke`, `gpu_interop_smoke`, `gpu_metal_smoke` with `--strict`.
  - All above jobs confirm effective CUDA path (no fallback warning/event).
- Status:
  - Implemented backend evidence artifacts and hardware-strict no-fallback gate in `scripts/run_verifylab_matrix.py`.
  - Added hardware-strict presets: `gpu_smoke_hw`, `gpu_interop_smoke_hw`, `gpu_metal_smoke_hw`.

### [x] PR-H02: CUDA-aware MPI strict A/B verification
- Risk target: R2
- Scope:
  - Add strict preset(s) for TD-MPI A/B runs:
    - baseline: `comm_overlap_isend=false`
    - overlap: `comm_overlap_isend=true`
    - optional cuda-aware variant: `cuda_aware_mpi=true`
  - Collect deterministic overlap artifacts (`csv/md`) per run.
- DoD:
  - 2-rank and 4-rank strict runs pass.
  - No invariant regressions.
  - Overlap benchmark artifact is produced for every CI run.
- Status:
  - Added strict presets in `scripts/run_verifylab_matrix.py`:
    - `mpi_overlap_smoke`
    - `mpi_overlap_cudaaware_smoke`
  - Presets run overlap A/B for ranks 2 and 4 and persist per-rank benchmark artifacts in run directory.
  - `scripts/bench_mpi_overlap.py` now enforces strict invariant counter checks (`hG`, `hV`, `violW`, `lagV`) by default.

### [x] PR-H03: Guardrail hardening for strict pipelines
- Risk target: R4
- Scope:
  - Promote critical warnings (zone geometry/runtime contract) to hard errors in strict mode/presets.
- DoD:
  - Strict presets fail on invalid zone width/runtime contract.
  - Diagnostic presets may keep warning behavior for analysis.
- Status:
  - `scripts/run_verifylab_matrix.py` enables strict guardrails under `--strict` (`strict_min_zone_width=True` in verification calls).
  - Added coverage in `tests/test_verifylab_guardrails.py`:
    - strict mode raises on invalid zone width,
    - diagnostic mode keeps warning behavior and returns rows.
  - Strict task-mode smoke presets now use geometry-valid `zones_total`.

### [x] PR-H04: Remove MPI uniform-mass task limitation (if semantically safe)
- Risk target: R5
- Scope:
  - Extend TD-MPI task path to support non-uniform masses for alloys.
  - Preserve existing TD scheduling semantics and invariants.
- DoD:
  - New tests for multi-mass alloy tasks in TD-MPI path.
  - Strict VerifyLab parity vs CPU reference remains within thresholds.
- Status:
  - `tdmd.main run --task ... --mode td_full_mpi` no longer enforces `require_uniform_mass`; per-atom masses are forwarded to `run_td_full_mpi_1d`.
  - `tdmd/state.py::kinetic_energy` now supports scalar or per-atom mass arrays, preserving TD-MPI thermo output under non-uniform masses.
  - Added CLI-level regression test (`tests/test_main_task_td_mpi_mass.py`) for non-uniform-mass alloy task handoff to TD-MPI runtime.
  - VerifyLab metal presets now use `examples/interop/task_eam_alloy.yaml` (non-uniform alloy masses).

### [x] PR-H05: Materials scientific parity pack (LAMMPS references)
- Risk target: R3
- Scope:
  - Add a curated reference suite (metals/alloys) with property-level checks.
  - Validate force/energy/temperature/pressure and trajectory metrics.
- DoD:
  - Strict thresholds are documented per case.
  - Reproducible reference fixtures available under versioned examples/results.
- Status:
  - Added versioned fixture: `examples/interop/materials_parity_suite_v1.json`.
  - Added strict checker: `scripts/materials_parity_pack.py` (force/energy/virial/T/P + trajectory + sync metrics).
  - Added regression test: `tests/test_materials_parity_pack.py`.
  - Documented per-case thresholds and run command in `docs/VERIFYLAB.md`.

### [x] PR-H06: Long-run stability envelope suite
- Risk target: R6
- Scope:
  - Introduce long-horizon strict checks for drift envelopes (`final_*`, `rms_*`).
- DoD:
  - Envelope baselines defined and enforced in CI-grade preset(s).
  - Regression detection is deterministic and documented.
- Status:
  - Added strict preset `longrun_envelope_ci` in `scripts/run_verifylab_matrix.py` (steps=300, rows for `zones_total={4,8}`).
  - Added baseline file `golden/longrun_envelope_v1.json` with per-row limits for `final_*` and `rms_*` metrics.
  - Implemented `apply_envelope_gate` in VerifyLab runner and persisted envelope diagnostics into run artifacts.
  - Added unit coverage `tests/test_verifylab_envelope_gate.py` and documented gate/threshold policy in `docs/VERIFYLAB.md`.

### [x] PR-H07: Failure observability and triage standardization
- Risk target: R7
- Scope:
  - Unify debug dumps (`td_trace`, invariants, backend selection, overlap counters).
  - Add one-command incident bundle export.
- DoD:
  - Any strict failure generates a reproducible diagnostic bundle.
  - Bundle format is documented for contributors.
- Status:
  - Added diagnostics module `tdmd/incident_bundle.py` for deterministic bundle creation with checksums and manifest.
  - `scripts/run_verifylab_matrix.py` now auto-creates `incident_bundle` on strict failures and records metadata in summaries.
  - Added one-command export tool `scripts/export_incident_bundle.py` (optional zip output).
  - Added tests in `tests/test_incident_bundle.py` and documented bundle format in `docs/VERIFYLAB.md`.

### [x] PR-H08: Contract consolidation in docs and governance
- Risk target: cross-cutting
- Scope:
  - Align all strategy/spec/verify docs with post-hardening contracts.
  - Clarify what is guaranteed in each mode (`serial`, `td_local`, `td_full_mpi`, gpu/multi-gpu).
- DoD:
  - Docs are internally consistent and reference strict gates by name.
- Status:
  - Added unified mode-contract governance doc: `docs/MODE_CONTRACTS.md` (mode guarantees, non-guarantees, strict gate mapping, triage contract).
  - Aligned strategy/spec/verify docs to reference post-hardening strict gates (`*_hw`, overlap strict, guardrails, materials parity, longrun envelope, incident bundle).
  - Updated `AGENTS.md` acceptance/gating policy to include post-hardening verification and governance contracts.

## Execution Template For Every PR-H0X
1. Pick one item in critical order.
2. Implement minimal PR-sized change.
3. Run mandatory gates:
   - `.venv/bin/python -m pytest -q`
   - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict`
   - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --strict`
   - If GPU touched:
     - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_smoke --strict`
4. Additional gates by scope:
   - Materials touched:
     - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset metal_smoke --strict`
     - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_metal_smoke --strict`
   - MPI overlap/cuda-aware touched:
     - `.venv/bin/python scripts/bench_mpi_overlap.py --config examples/td_1d_morse_static_rr_smoke4.yaml --n 4 --overlap-list 0,1 --out results/mpi_overlap.csv --md results/mpi_overlap.md`
5. Update docs (minimum):
   - `docs/SPEC_TD_AUTOMATON.md`
   - `docs/INVARIANTS.md`
   - `docs/THEORY_VS_CODE.md`
   - `docs/VERIFYLAB.md`
   - GPU-related: `docs/GPU_BACKEND.md`, `docs/VERIFYLAB_GPU.md`

## Exit Criteria (Risk Burndown Complete)
- No unresolved High severity risks (R1-R3) without explicit waiver.
- Strict GPU lane confirms effective CUDA path (no-fallback acceptance).
- TD-MPI overlap/cuda-aware path is strictly verified on 2/4 ranks.
- Material parity suite and long-run envelopes are stable.
- Governance/docs fully aligned to enforced behavior.
