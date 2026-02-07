# Theory vs Code Mapping

This document maps dissertation terms to their current code locations.
Mode-level guarantees and strict gate mapping are documented in `docs/MODE_CONTRACTS.md`.

## Core TD Automaton
- **States F/D/P/W/S**: `tdmd/zones.py` (`ZoneType`), `tdmd/td_automaton.py` (`TDAutomaton1W`)
- **Zone runtime state** (`A(z)`, halo, step_id): `tdmd/td_automaton.py` (`ZoneRuntime`)
- **Traversal order / fairness**: `tdmd/td_full_mpi.py` (`traversal_order`), `tdmd/td_local.py` (order)
- **Formal core constraint (≤1 zone in W)**: `tdmd/td_automaton.py::_assert_invariants`

## Dependencies & Support
- **table-deps / owner-deps**: `tdmd/td_full_mpi.py` (`deps_zone_ids`, `owner_deps_zone_ids`, `set_deps_funcs/preds`)
- **DepsProvider abstraction**: `tdmd/deps_provider.py`, `tdmd/deps_provider_3d.py`
- **support(table(z))**: `tdmd/interaction_table_state.py` (`support_ids`), `tdmd/zone_bins_localz.py::support_ids`
- **Candidate set for tables**: `tdmd/td_automaton.py::ensure_table`

## Task IO Runtime Contract
- **Task schema parse/load**: `tdmd/io/task.py` (`parse_task_dict`, `load_task`)
- **Run-time compatibility gate**: `tdmd/io/task.py::validate_task_for_run`
- **Ensemble contract**: `task.ensemble.kind in {nve,nvt,npt}` with strict shape checks
  (`nvt` requires thermostat, `npt` requires thermostat+barostat, `nve` forbids both).
- **Contract document**: `docs/ENSEMBLES.md`
- **Current runtime status**:
  - `serial`: `nve/nvt/npt`
  - `td_local`: `nve/nvt/npt`
  - `td_full_mpi`: `nve/nvt`, plus `npt` when `MPI size=1`
- **Legacy alias**: top-level `thermostat` is parsed for backward compatibility and remains non-executable.
- **Ensemble runtime controllers**: `tdmd/ensembles.py` (`berendsen` thermostat/barostat), called from
  `tdmd/serial.py`, `tdmd/td_local.py`, and `tdmd/td_full_mpi.py`.
- **Interop-only task fields (for now)**: charged atoms
- **Run-time constraints (current)**: periodic `pbc` in all directions, cubic box, strict potential params set
- **Type-dependent pair matrix**: `potential.params.pair_coeffs` (e.g. `1-2`) for `lj`/`morse`; validated as a complete matrix for task atom types
- **Tabulated pair potential runtime**: `potential.kind=table` (`file` + `keyword`) is executed by `make_potential -> TablePotential` and used by force kernels in `serial`/`td_local`
- **EAM/alloy CPU reference**:
  - `potential.kind=eam/alloy` (`eam_alloy` alias) is parsed via DYNAMO setfl reader and executed in `serial` by `EAMAlloyPotential` (two-pass density+force)
  - type-to-element mapping comes from `potential.params.elements` (`type t -> elements[t-1]`)
- **EAM in TD-local**:
  - `td_local` supports many-body backend calls via `EAMAlloyPotential.forces_energy_virial` (zone-wise integration path preserved)
  - verification parity path uses `sync_mode=True` for serial-equivalent snapshot updates
- **EAM in TD-MPI (`td_full_mpi`)**:
  - `td_automaton.compute_step_for_work_zone` uses `potential.forces_on_targets(...)` when available (many-body support without automaton state changes)
  - EAM target-zone forces use table support candidate set (`table.support_ids`) plus local zone ids
- **Mass/type support (pair runtime)**:
  - `serial` / `td_local`: multi-type and multi-mass are supported for pair models
  - `td_full_mpi`: task path supports multi-type and multi-mass (runtime passes per-atom mass array to TD-MPI integrator path)
- **EAM mode gates**:
  - `serial` task mode: `lj/morse/table/eam/alloy`
  - `td_local` verify path: `lj/morse/table/eam/alloy`
  - `td_full_mpi` task mode: `lj/morse/table/eam/alloy` with non-uniform per-atom masses supported
- **Config ensemble gate**:
  - `tdmd/config.py` parses `ensemble` with strict validation.
  - mode-specific runtime guardrails are enforced in execution paths (`serial`, `td_local`, `td_full_mpi`).
  - `td_full_mpi` currently guards `npt` to single-rank launch.
- **LAMMPS interop for metals/alloys**:
  - `tdmd/io/lammps.py::export_lammps_in` supports `pair_style eam/alloy` and emits
    `pair_coeff * * <setfl> <elem(type=1)> ... <elem(type=N)>` from `potential.params.elements`.
  - `tdmd/io/lammps.py::export_lammps_in` emits `fix nve/nvt/npt` according to `task.ensemble`.
  - `tdmd/io/lammps.py::export_lammps_data` enforces LAMMPS-compatible contiguous atom types (`1..N`) for exported data files.

## Halo / Migration
- **Migration DELTA**: `tdmd/td_full_mpi.py` (`REC_DELTA`, `DELTA_MIGRATION`)
- **HALO (ghosts)**: `tdmd/td_full_mpi.py` (`DELTA_HALO`, `zone.halo_ids`), `tdmd/td_automaton.py` (halo usage in `ensure_table`)
- **HALO geometry checks**: `tdmd/td_full_mpi.py` (`hG`), `tdmd/td_automaton.py` (`hG3/hV3` for 3D)
- **Outbox/holder routing**: `tdmd/td_full_mpi.py` (`holder_map`, `REC_HOLDER`, `REQ_*`)

## Time Decomposition Semantics
- **zone time layer** (`step_id`): `tdmd/td_automaton.py`, `tdmd/td_full_mpi.py`
- **time-lag safety** (`max_step_lag`): `tdmd/td_automaton.py::_lag_ok`, `tdmd/td_full_mpi.py` (lag gating)
- **buffer/skin vs lag**: `tdmd/zones.py::compute_zone_buffer_skin`, `tdmd/td_full_mpi.py::update_buffers`

## Geometry / PBC
- **1D slab ranges**: `tdmd/zones.py::zones_overlapping_range_pbc`
- **3D AABB/PBC**: `tdmd/geom_pbc.py`, `tdmd/zones.py::zones_overlapping_aabb_pbc`
- **3D invariants (I3/I4)**: `tdmd/td_automaton.py` (`tG3/hG3/hV3`), `tdmd/td_full_mpi.py`

## Verification & Golden Baselines
- **Serial reference**: `tdmd/serial.py`
- **TD-local reference**: `tdmd/td_local.py` (async + `sync_mode`)
- **VerifyLab sweeps**: `tdmd/verify_lab.py`, `scripts/run_verifylab_matrix.py`
- **Strict CI smoke split**: `scripts/run_verifylab_matrix.py` presets `smoke_ci` (strict gate) and `smoke_regression` (diagnostic profile)
- **Materials VerifyLab presets**:
  - `metal_smoke`: strict task-based EAM smoke (`examples/interop/task_eam_al.yaml`, sync verification mode),
  - `interop_metal_smoke`: strict alloy interop EAM smoke (`examples/interop/task_eam_alloy.yaml`, sync verification mode).
- **Time-blocking formalization (`K>1`)**:
  - `td.time_block_k` is parsed as an alias of `td.batch_size` in `tdmd/config.py` and forwarded to `td_full_mpi` send batching.
  - batching diagnostics are tracked in `tdmd/td_automaton.py` (`send_batches`, `send_batch_zones_total`, `send_batch_size_max`)
    and exposed in TD-MPI periodic logs (`tbK`, `sb`, `sbAvg`, `sbMax`).
- **Backend abstraction (GPU track stage G01)**:
  - `tdmd/backend.py::resolve_backend` implements `run.device` selection (current: `auto|cpu|cuda`) with safe CPU fallback.
  - `tdmd.main run --device ...` overrides config device selection at runtime.
  - backend selection does not alter TD automaton behavior; CPU remains reference semantics.
  - planned portability extension (NVIDIA + AMD via Kokkos) is tracked in `docs/PORTABILITY_KOKKOS_PLAN.md`; this is a backend-path extension only and does not alter TD semantics.
- **GPU pair-kernel path (GPU track stage G02)**:
  - `tdmd/forces_gpu.py::forces_on_targets_pair_backend` implements CUDA force kernels for `LJ/Morse/table/eam-alloy`.
  - `serial` and `td_local` call the GPU pair kernel only for pair potentials and only with identical target/candidate semantics as CPU path.
  - when CUDA backend is unavailable, execution falls back to CPU path without semantic change.
- **GPU neighbor/cell-list candidate path (GPU track stage G03)**:
  - `tdmd/forces_gpu.py::forces_on_targets_celllist_backend` reuses CPU-equivalent cell-list candidate construction and evaluates pruned candidate interactions on GPU.
  - this preserves CPU candidate-set semantics while reducing interaction workload versus all-pairs GPU evaluation.
- **MPI overlap transport refinement (GPU track stage G08)**:
  - `tdmd/td_full_mpi.py` supports optional nonblocking send path (`td.comm_overlap_isend`) and optional CUDA-aware transport toggle (`td.cuda_aware_mpi`) with host-staging fallback.
  - benchmark helper `scripts/bench_mpi_overlap.py` compares blocking and overlap transport modes and writes reproducible artifacts.
- **Hardware-strict GPU verification gate (risk burndown PR-H01)**:
  - `scripts/run_verifylab_matrix.py` resolves backend evidence (`requested_device`, `effective_device`, fallback) and writes it into run artifacts.
  - hardware-strict presets (`*_hw`) and `--require-effective-cuda` fail verification when CUDA request resolves to CPU fallback.
  - planned policy generalization to `--require-effective-gpu` is portability-cycle verification policy only.
- **TD-MPI overlap strict presets (risk burndown PR-H02)**:
  - `scripts/run_verifylab_matrix.py` includes `mpi_overlap_smoke` and `mpi_overlap_cudaaware_smoke` presets.
  - each preset invokes `scripts/bench_mpi_overlap.py` for 2/4 ranks, stores per-rank overlap artifacts, and enforces invariant-counter gate (`hG/hV/violW/lagV`).
- **Strict guardrail hardening (risk burndown PR-H03)**:
  - `scripts/run_verifylab_matrix.py` enables strict geometry guardrails under `--strict` (`strict_min_zone_width=True` in TD-local verification calls).
  - task-based strict presets use geometry-valid `zones_total` to avoid warning-only behavior in strict CI.
- **TD-MPI task multi-mass support (risk burndown PR-H04)**:
  - `tdmd.main run --task ... --mode td_full_mpi` no longer enforces `require_uniform_mass`; per-atom masses are forwarded to `run_td_full_mpi_1d`.
  - `tdmd/state.py::kinetic_energy` supports both scalar and per-atom mass arrays, preserving TD-MPI thermo reporting.
- **Materials parity pack (risk burndown PR-H05)**:
  - `scripts/materials_parity_pack.py` validates versioned materials fixture
    `examples/interop/materials_parity_suite_v1.json` with strict per-case thresholds.
  - validation covers force/energy/virial/T/P plus serial trajectory endpoint and serial-vs-`td_local(sync)` metrics.
- **Long-run envelope gate (risk burndown PR-H06)**:
  - `scripts/run_verifylab_matrix.py` preset `longrun_envelope_ci` applies strict baseline checks from `golden/longrun_envelope_v1.json`.
  - baseline enforces long-horizon `final_*` and `rms_*` drift envelopes without changing TD semantics.
- **Failure observability / incident bundle (risk burndown PR-H07)**:
  - `tdmd/incident_bundle.py` and `scripts/export_incident_bundle.py` provide one-command diagnostic bundle export.
  - `scripts/run_verifylab_matrix.py` auto-generates `incident_bundle` on strict failures and persists metadata in run summaries.
- **Cluster validation lanes (risk burndown v2 PR-V2-01..PR-V2-04)**:
  - profile contract + loader: `tdmd/cluster_profile.py`,
  - overlap benchmark with normalized artifacts: `scripts/bench_mpi_overlap.py`,
  - strong/weak scaling strict gate: `scripts/bench_cluster_scale.py`,
  - long-run cluster stability strict gate: `scripts/bench_cluster_stability.py`,
  - transport/fabric matrix strict gate: `scripts/bench_mpi_transport_matrix.py`,
  - VerifyLab presets: `cluster_scale_smoke`, `cluster_stability_smoke`, `mpi_transport_matrix_smoke`.
- **Materials property-reference expansion (risk burndown v2 PR-V2-05..PR-V2-07)**:
  - expanded fixtures/tasks: `examples/interop/materials_parity_suite_v2*.json`, `examples/interop/task_eam_*_t*.yaml`,
  - fixture generator: `scripts/generate_materials_parity_fixture.py`,
  - parity checker with by-property (`eos/thermo/transport`) summaries: `scripts/materials_parity_pack.py`,
  - strict property gate wrapper: `scripts/materials_property_gate.py`,
  - threshold calibration artifact: `scripts/calibrate_material_thresholds.py`, `golden/material_threshold_policy_v2.json`.
- **Golden checks**: `tdmd/verify_golden.py`, `tests/test_golden*.py`

## Visualization / Observability Boundary
- **Trajectory writer**: `tdmd/io/trajectory.py`
  - LAMMPS dump-compatible streaming output,
  - optional channels: `xu/yu/zu`, `ix/iy/iz`, `fx/fy/fz`,
  - optional gzip compression and versioned sidecar manifest.
- **Global metrics writer**: `tdmd/io/metrics.py`
  - deterministic time-series observables export with sidecar manifest.
- **Manifest governance**: `tdmd/io/manifest.py`
  - schema/version metadata for trajectory and metrics contracts.
- **Output wiring**: `tdmd/output.py`, `tdmd/main.py`, `tdmd/td_full_mpi.py`
  - unified CLI/runtime knobs for `serial`, `td_local`, `td_full_mpi`.
- **Post-processing API**: `tdmd/viz.py`
  - mode-agnostic trajectory iterator and plugin execution pipeline.
- **Adapters/bundle**: `scripts/viz_analyze.py`, `scripts/viz_bundle.py`, `scripts/viz_*_adapter.py`
  - external OVITO/VMD/ParaView integration is script-driven and reproducible.
- **Strict verification lane**: `scripts/bench_viz_contract.py` + VerifyLab preset `viz_smoke`.
- **Contract rule**: visualization/output is passive and must not affect TD/runtime decisions.
- **Governance document**: `docs/VISUALIZATION.md`.
