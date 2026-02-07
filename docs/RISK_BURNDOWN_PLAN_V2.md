# Risk Burndown Plan v2 (Cluster + Materials Properties)

This cycle starts after completion of `docs/RISK_BURNDOWN_PLAN.md` (`PR-H01..PR-H08`).
Status: completed (`PR-V2-01..PR-V2-08`).
Next active cycle is tracked separately in `docs/PORTABILITY_KOKKOS_PLAN.md` (`PR-K01..PR-K10`).

## Goal
Reduce the next high-impact risks for production-grade metals/alloys TDMD runs:
- cluster performance and stability validation under strict, reproducible conditions,
- expanded property-level materials references across broader thermodynamic states.

## Non-Negotiable Rules
- Keep TD states/transitions unchanged (`F/D/P/W/S`) unless explicitly formalized and verified.
- Do not introduce implicit global barriers.
- CPU path remains formal reference semantics; GPU is refinement only.
- Any behavior change requires tests + strict VerifyLab + docs updates.
- Hardware-strict GPU tasks cannot be accepted with CPU fallback.

## Current Risk Register (v2)
| ID | Risk | Severity | Likelihood | Why it matters |
|---|---|---:|---:|---|
| C1 | Cluster scaling claims are not reproducible across node counts/topologies | High | Medium | optimization decisions can become non-portable or misleading |
| C2 | Long-run cluster stability under async/overlap is under-validated | High | Medium | latent hangs/drift can appear only in long production runs |
| C3 | MPI overlap/cuda-aware paths are not fully characterized per fabric | High | Medium | transport-level regressions can pass local smoke and fail on clusters |
| M1 | Property-level material references are too narrow in T/rho/composition | High | High | weak scientific confidence for alloys domain |
| M2 | Materials thresholds are not structured by property sensitivity | Medium | Medium | false negatives/positives in strict gates |
| M3 | Reference fixture provenance/versioning is incomplete at scale | Medium | Medium | difficult reproducibility and auditability |

## Critical Order (PR-sized)

### [x] PR-V2-01: Cluster benchmark contract and reproducibility profile
- Risk target: C1, C3
- Scope:
  - Define cluster benchmark profile schema (node count, ranks, GPUs, fabric, overlap/cuda-aware flags, pinning policy).
  - Standardize artifact bundle for cluster runs (`config + env + metrics + incident hooks`).
- DoD:
  - New documented profile contract and fixture examples.
  - Bench scripts accept profile file and persist normalized artifacts.
- Status:
  - Added cluster profile contract/parser in `tdmd/cluster_profile.py`.
  - Added profile fixtures:
    - `examples/cluster/cluster_profile_smoke.yaml` (CI/simulated smoke),
    - `examples/cluster/cluster_profile_real_template.yaml` (hardware strict template).
  - `scripts/bench_mpi_overlap.py` now supports profile-driven execution, normalized summary JSON artifacts, env snapshot capture, and controlled simulated mode.

### [x] PR-V2-02: Strict strong/weak scaling verification lane
- Risk target: C1
- Scope:
  - Add strict VerifyLab presets for strong/weak scaling envelopes on cluster runners.
  - Gate speedup/efficiency against documented baseline envelopes.
- DoD:
  - Strict preset(s) run for at least 2 node scales (for configured cluster target).
  - Regressions fail deterministically with artifact evidence.
- Status:
  - Added `scripts/bench_cluster_scale.py` with strict strong/weak gates and reproducible `csv/md/json` artifacts.
  - Added VerifyLab strict preset `cluster_scale_smoke`.
  - Added envelope baseline `golden/cluster_scale_envelope_v1.json` and profile linkage (`metadata.scale_envelope_file`).

### [x] PR-V2-03: Cluster long-run stability suite
- Risk target: C2
- Scope:
  - Add long-horizon cluster stability presets (async + overlap variants).
  - Gate drift/invariant counters/lag counters under extended steps.
- DoD:
  - Strict long-run cluster suite passes with explicit envelope thresholds.
  - Any failure auto-produces incident bundle with checksums.
- Status:
  - Added `scripts/bench_cluster_stability.py` (long-run blocking/overlap variants with invariant/diag thresholds and strict artifacts).
  - Added VerifyLab strict preset `cluster_stability_smoke`.
  - Strict failures in this lane inherit existing incident-bundle auto-export from `scripts/run_verifylab_matrix.py`.

### [x] PR-V2-04: MPI transport matrix hardening (fabric-aware)
- Risk target: C3
- Scope:
  - Extend overlap/cuda-aware A/B to transport matrix (host-staging vs cuda-aware, overlap on/off, configured fabric profiles).
  - Add deterministic comparison report for communication cost and invariants.
- DoD:
  - Strict matrix runs pass on cluster profile set.
  - Report artifacts (`csv` + `md`) generated for every strict run.
- Status:
  - Added `scripts/bench_mpi_transport_matrix.py` (transport profile matrix by rank/fabric/overlap/cuda-aware with strict checks).
  - Added VerifyLab strict preset `mpi_transport_matrix_smoke`.
  - Matrix artifacts are persisted per run (`csv/md/json`) and summarized in VerifyLab summary artifacts.

### [x] PR-V2-05: Materials parity suite v2 (state-space expansion)
- Risk target: M1, M3
- Scope:
  - Expand materials fixtures to more cases/temperatures/densities/compositions.
  - Keep references versioned and provenance-tracked.
- DoD:
  - New fixture pack (v2) includes broader T/rho coverage and alloy mixes.
  - Strict parity script validates all cases reproducibly.
- Status:
  - Added expanded state-space tasks:
    - `examples/interop/task_eam_al_t600.yaml`,
    - `examples/interop/task_eam_al_dense_t600.yaml`,
    - `examples/interop/task_eam_alloy_t900.yaml`,
    - `examples/interop/task_eam_alloy_dense_t900.yaml`.
  - Added v2 parity fixtures:
    - template: `examples/interop/materials_parity_suite_v2_template.json`,
    - generated reference: `examples/interop/materials_parity_suite_v2.json`.
  - Added reproducible fixture generator: `scripts/generate_materials_parity_fixture.py`.

### [x] PR-V2-06: Property-level gates (EOS/thermo/transport subset)
- Risk target: M1, M2
- Scope:
  - Add explicit property-level checks (e.g., EOS points, thermo consistency, selected transport proxies) with per-property tolerances.
  - Integrate with VerifyLab strict preset(s).
- DoD:
  - Property gate outputs per-property pass/fail with threshold provenance.
  - CI-grade strict preset fails on property regressions.
- Status:
  - Extended `scripts/materials_parity_pack.py` with explicit property-group checks (`eos`, `thermo`, `transport`) and `by_property` summary.
  - Added strict property gate wrapper `scripts/materials_property_gate.py`.
  - Added VerifyLab strict presets:
    - `metal_property_smoke`,
    - `interop_metal_property_smoke`.

### [x] PR-V2-07: Threshold governance and sensitivity calibration
- Risk target: M2
- Scope:
  - Formalize threshold rationale by property class and thermodynamic regime.
  - Add calibration notebook/script outputs to versioned artifacts.
- DoD:
  - Threshold policy documented and linked to fixture provenance.
  - Regression tests protect policy shape and parsing.
- Status:
  - Added calibration tool `scripts/calibrate_material_thresholds.py`.
  - Added versioned threshold policy artifact `golden/material_threshold_policy_v2.json` and run report `results/material_threshold_policy_v2.md`.
  - Added tests for policy shape and script execution (`tests/test_material_threshold_calibration.py`).

### [x] PR-V2-08: v2 contract consolidation and ops handoff
- Risk target: cross-cutting
- Scope:
  - Consolidate cluster/materials-property contracts into governance docs.
  - Publish execution playbook for contributors and CI/cluster operators.
- DoD:
  - Docs are internally consistent and cross-referenced.
  - All active strict gates are listed by scope and ownership.
- Status:
  - Consolidated v2 governance/docs and prompt/agent guidance (cluster + materials property lanes).
  - Added/updated strict gate documentation in VerifyLab and mode-contract governance.

## Execution Template For Every PR-V2-0X
1. Pick one item in critical order.
2. Implement minimal PR-sized change.
3. Run mandatory gates:
   - `.venv/bin/python -m pytest -q`
   - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict`
   - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --strict`
4. Additional gates by scope:
   - GPU touched:
     - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_smoke --strict`
   - Materials touched:
     - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset metal_smoke --strict`
     - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_metal_smoke --strict`
   - MPI overlap/cuda-aware touched:
     - `.venv/bin/python scripts/bench_mpi_overlap.py --config examples/td_1d_morse_static_rr_smoke4.yaml --n 4 --overlap-list 0,1 --out results/mpi_overlap.csv --md results/mpi_overlap.md`
   - Cluster scaling/stability touched (v2):
     - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset cluster_scale_smoke --strict`
     - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset cluster_stability_smoke --strict`
   - Materials property-level gates touched (v2):
     - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset metal_property_smoke --strict`
     - `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_metal_property_smoke --strict`
5. Update docs (minimum):
   - `docs/SPEC_TD_AUTOMATON.md`
   - `docs/INVARIANTS.md`
   - `docs/THEORY_VS_CODE.md`
   - `docs/VERIFYLAB.md`
   - GPU-related: `docs/GPU_BACKEND.md`, `docs/VERIFYLAB_GPU.md`
   - v2 governance changes: `docs/MODE_CONTRACTS.md`

## Exit Criteria (v2 Complete)
- Cluster strict lanes are reproducible across target scale profiles with enforced envelopes.
- Long-run cluster stability gates are green with deterministic artifacts.
- Materials parity suite v2 covers broader T/rho/composition space with provenance.
- Property-level strict gates are active and documented.
- Governance/docs are aligned with implemented strict behavior.
