# Roadmap

## v4.4
- VerifyLab presets + artifacts (smoke/paper/stress/async) — DoD: `scripts/run_verifylab_matrix.py` writes `results/<run_id>/{config.json,metrics.csv,summary.md}` for each preset.
- Smoke thresholds + CI test — DoD: `tests/test_verifylab_smoke.py` fails on any non-ok row; explicit dr/dv/dE/dT/dP thresholds are wired to smoke.
- Golden baselines — DoD: `golden/cfg_system_smoke.json` exists and `tests/test_golden_smoke.py` passes.
- Async drift visibility — DoD: metrics CSV includes `final_*` and `rms_*`; `verify_plots` renders these columns.
- Build/verify stabilization — DoD: `scripts/run_verifylab_matrix.py` runs from repo root, and import/py_compile smoke tests keep `tdmd/{td_automaton,td_full_mpi,main}.py` syntax-clean.

## v4.5
- Formal TD automaton spec — DoD: `docs/SPEC_TD_AUTOMATON.md` has full transition table with code references.
- Invariants map + checks — DoD: `docs/INVARIANTS.md` maps each invariant to concrete checks (incl. `tG3/hG3/hV3`) and at least two tests cover them.

## v4.6
- General deps-graph (non-ring) stabilization — DoD: `DepsProvider3D` tests + `docs/DEPSGRAPH.md` updated; TD-MPI uses owner/table deps without ring assumptions.

## v4.7 (Post-Baseline Hardening)
- Hardware-strict GPU verification lane (`*_hw`) — DoD: CUDA fallback is explicit failure in hardware-strict runs.
- MPI overlap/cuda-aware strict A/B gates — DoD: invariant-safe overlap artifacts for 2/4 ranks.
- Strict geometry guardrails — DoD: invalid zone geometry is fatal in strict mode.
- TD-MPI task multi-mass support — DoD: non-uniform alloy masses accepted in MPI task path.
- Materials parity suite — DoD: versioned fixture + strict property-level parity checks.
- Long-run envelope gate — DoD: deterministic `final_*`/`rms_*` drift baselines in strict preset.
- Incident bundle standardization — DoD: strict failures auto-produce reproducible diagnostic bundles.
- Contract consolidation — DoD: mode guarantees and strict-gate ownership documented in `docs/MODE_CONTRACTS.md`.

## v4.8 (Risk Burndown v2, Completed)
- Cluster reproducibility profile + benchmark contract — DoD met via `tdmd/cluster_profile.py`, profile fixtures, and normalized benchmark summaries.
- Strong/weak scaling strict lanes — DoD met via `scripts/bench_cluster_scale.py`, preset `cluster_scale_smoke`, and baseline envelope `golden/cluster_scale_envelope_v1.json`.
- Cluster long-run stability strict lanes — DoD met via `scripts/bench_cluster_stability.py` and preset `cluster_stability_smoke`.
- MPI transport matrix hardening — DoD met via `scripts/bench_mpi_transport_matrix.py` and preset `mpi_transport_matrix_smoke`.
- Materials parity suite v2 expansion — DoD met via expanded task set and `examples/interop/materials_parity_suite_v2.json`.
- Property-level strict materials gates — DoD met via `scripts/materials_property_gate.py` and presets `metal_property_smoke` / `interop_metal_property_smoke`.
- Threshold governance calibration — DoD met via `scripts/calibrate_material_thresholds.py` and artifact `golden/material_threshold_policy_v2.json`.
- v2 governance consolidation — DoD met via updates to strategy/spec/verify docs and active-mode contract mapping.

## v4.9 (Universal Visualization and Analysis, Completed)
- PR-VZ01 Output schema governance for trajectory/metrics manifests.
  - DoD met via sidecar schema/version manifests and compatibility tests.
- PR-VZ02 Runtime output abstraction (mode-agnostic writer interface).
  - DoD met for `serial`, `td_local`, and `td_full_mpi` output controls.
- PR-VZ03 Extended channels (`xu/yu/zu`, `ix/iy/iz`, `fx/fy/fz`) with explicit flags.
  - DoD met via deterministic channel/column contract tests.
- PR-VZ04 Universal post-processing API (`scripts/viz_*`) with plugin hooks.
  - DoD met via plugin runner and stable CSV/JSON artifacts.
- PR-VZ05 External adapters (OVITO/VMD/ParaView) with reproducible presets.
  - DoD met via scripted adapters and bundle orchestration under run artifacts.
- PR-VZ06 Large-scale output policy (chunking/compression/thinning).
  - DoD met via frame-thinning controls, streaming writers, and optional gzip output.
- PR-VZ07 Visualization strict gate integration.
  - DoD met via `viz_smoke` preset in `scripts/run_verifylab_matrix.py`.
- PR-VZ08 Documentation and workflow consolidation.
  - DoD met via updates to `README.md`, `docs/VERIFYLAB.md`, prompts, and governance docs.

## v5.0 (GPU Portability via Kokkos, Planned)
- PR-K01 Governance freeze for portability cycle.
  - DoD: PR order, strict-gate mapping, and prompt/doc ownership frozen in `docs/PORTABILITY_KOKKOS_PLAN.md`.
- PR-K02 Backend selector contract expansion (`auto|cpu|cuda|hip|kokkos`).
  - DoD: control-plane compatibility with safe fallback and strict evidence fields.
- PR-K03 VerifyLab hardware-strict generalization (`require_effective_gpu` policy).
  - DoD: hardware-strict no-fallback policy is vendor-neutral and artifact-backed.
- PR-K04 Kokkos runtime bridge/build integration.
  - DoD: deterministic runtime capability detection for NVIDIA/AMD backends.
- PR-K05 Pair kernels (LJ/Morse) on Kokkos backend.
  - DoD: strict CPU-parity in `serial`/`td_local`.
- PR-K06 Table + EAM/eam-alloy kernels on Kokkos backend.
  - DoD: strict materials gates and parity packs stay green.
- PR-K07 td_local Kokkos hardening.
  - DoD: sync/async contracts preserved; no invariant regressions.
- PR-K08 td_full_mpi vendor-neutral multi-GPU path.
  - DoD: overlap/transport lanes green without new global barriers.
- PR-K09 Vendor strict lanes + cross-vendor parity governance.
  - DoD: NVIDIA/AMD hardware-strict lanes reject fallback and publish backend evidence.
- PR-K10 Final consolidation and ops handoff.
  - DoD: docs/prompts/gates are internally consistent and auditable.
