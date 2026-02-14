# VerifyLab

CUDA planning reference: `docs/CUDA_EXECUTION_PLAN.md` (PR-C01..PR-C08).

How to run:
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict
```
Long-run strict envelope gate:
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset longrun_envelope_ci --strict
```
Manual incident bundle export (one command):
```bash
python scripts/export_incident_bundle.py --run-dir results/<run_id> --reason manual_export --zip-out results/<run_id>/incident_bundle.zip
```
Task interop smoke:
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --task examples/interop/task.yaml --strict
```
Alloy pair-matrix interop smoke:
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --task examples/interop/task_alloy_pair.yaml --strict
```
Table-potential interop smoke:
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --task examples/interop/task_table.yaml --strict
```
Ensemble strict smokes:
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset nvt_smoke --strict
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset npt_smoke --strict
```
EAM serial sample run (reference backend):
```bash
python -m tdmd.main run --task examples/interop/task_eam_al.yaml --mode serial
python -m tdmd.main run --task examples/interop/task_eam_alloy.yaml --mode serial
```
Open-library EAM sample run (`Fe/Ni/Ti`):
```bash
python -m tdmd.main run --task examples/interop/task_eam_fe_mishin2006.yaml --mode serial
python -m tdmd.main run --task examples/interop/task_eam_ni99.yaml --mode serial
python -m tdmd.main run --task examples/interop/task_eam_ti_zhou04.yaml --mode serial
```
EAM reference parity test (synthetic LAMMPS-compatible fixture):
```bash
python -m pytest -q tests/test_eam_runtime.py::test_eam_matches_lammps_reference_fixture
```
EAM serial-vs-td_local parity tests:
```bash
python -m pytest -q tests/test_eam_td_local.py
```
EAM automaton/MPI-path unit check (no MPI launch required):
```bash
python -m pytest -q tests/test_td_automaton_eam.py
```
EAM/metal strict smoke presets:
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset metal_smoke --strict
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_metal_smoke --strict
```
Materials parity pack (versioned fixture, strict property-level checks):
```bash
python scripts/materials_parity_pack.py --fixture examples/interop/materials_parity_suite_v1.json --config examples/td_1d_morse.yaml --strict --out results/materials_parity_suite_v1_summary.json
python scripts/materials_parity_pack.py --fixture examples/interop/materials_parity_suite_v2.json --config examples/td_1d_morse.yaml --strict --out results/materials_parity_suite_v2_summary.json
```
Materials property-level strict gates (v2):
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset metal_property_smoke --strict
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_metal_property_smoke --strict
```
Cluster v2 strict lanes (profile-driven):
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset cluster_scale_smoke --strict
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset cluster_stability_smoke --strict
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset mpi_transport_matrix_smoke --strict
```
Visualization contract strict lane:
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset viz_smoke --strict
```

## CI Smoke
- CI/unit tests use the config-defined random system (`cfg_system`) for a **single-step** serial vs TD-local check.
- Strict CI gate:
  `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict`
- Regression/diagnostic smoke (legacy profile, non-strict by default):
  `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_regression`
- Backward-compatible alias:
  `--preset smoke` is an alias for `smoke_regression`.

## MPI Smoke (static_rr)
Manual run (requires `mpirun` + `mpi4py`):
```bash
python scripts/run_mpi_smoke.py --n 2 --config examples/td_1d_morse_static_rr.yaml
python scripts/run_mpi_smoke.py --n 4 --config examples/td_1d_morse_static_rr_smoke4.yaml
```

Pytest hook (skipped unless enabled):
```bash
TDMD_MPI_SMOKE=1 python -m pytest -q tests/test_mpi_smoke.py
```

Time-blocking benchmark (`K>1`, requires `mpirun`):
```bash
python scripts/bench_time_blocking.py --config examples/td_1d_morse_static_rr.yaml --n 2 --k-list 1,2,4 --steps 40 --out results/time_blocking.csv
```

MPI overlap benchmark (`comm_overlap_isend`, optional `cuda_aware_mpi`, requires `mpirun`):
```bash
python scripts/bench_mpi_overlap.py --config examples/td_1d_morse_static_rr_smoke4.yaml --n 4 --overlap-list 0,1 --cuda-aware --out results/mpi_overlap.csv --md results/mpi_overlap.md
```
Profile-driven overlap benchmark (normalized artifacts + env snapshot):
```bash
python scripts/bench_mpi_overlap.py --profile examples/cluster/cluster_profile_smoke.yaml --out results/mpi_overlap_profile.csv --md results/mpi_overlap_profile.md
```
Cluster scale/stability/transport scripts:
```bash
python scripts/bench_cluster_scale.py --profile examples/cluster/cluster_profile_smoke.yaml --strict
python scripts/bench_cluster_stability.py --profile examples/cluster/cluster_profile_smoke.yaml --strict
python scripts/bench_mpi_transport_matrix.py --profile examples/cluster/cluster_profile_smoke.yaml --strict
```
Notes:
- `examples/cluster/cluster_profile_smoke.yaml` is CI-safe and allows simulated execution when MPI launcher/hardware is unavailable.
- Hardware-cluster strict runs should use `examples/cluster/cluster_profile_real_template.yaml` (or derived profile) with simulated fallback disabled.

MPI overlap strict presets (A/B at 2 and 4 ranks):
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset mpi_overlap_smoke --strict
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset mpi_overlap_cudaaware_smoke --strict
```

Dry-run (no MPI launch):
```bash
python scripts/bench_time_blocking.py --config examples/td_1d_morse_static_rr.yaml --k-list 1,2,4 --dry-run
```

## Sync Mode (td_local)
- `run_td_local(..., sync_mode=True)` runs a **synchronous snapshot** update for verification.
  It is intended for long-step equivalence checks (serial vs TD-local) without TD asynchrony.

## Metrics (VerifyLab CSV)
For each row in `metrics.csv`:
- `max_dr/max_dv`: max atom-wise drift at the **final step** (serial vs TD-local).
- `max_dE/max_dT/max_dP`: max absolute observable drift across **all observer steps**.
- `final_dr/final_dv`: max atom-wise drift at the **last observer step**.
- `final_dE/final_dT/final_dP`: observable drift at the **last observer step**.
- `rms_dr/rms_dv`: RMS of max atom-wise drift over **observer steps**.
- `rms_dE/rms_dT/rms_dP`: RMS of observable drift over **observer steps**.

## Presets
- `smoke_ci`: short strict pass/fail gate for CI.
- `smoke_regression`: regression/diagnostic sweep; use with/without `--strict` depending on goal.
- `smoke`: backward-compatible alias for `smoke_regression`.
- `interop_smoke`: serial vs TD-local on an explicit task file (LAMMPS interop check).
- `nvt_smoke`: strict task-based NVT parity smoke (`task_nvt.yaml`, sync mode).
- `npt_smoke`: strict task-based NPT parity smoke (`task_npt.yaml`, sync mode).
- `metal_smoke`: strict EAM task-based smoke (`task_eam_al.yaml`, sync mode).
- `interop_metal_smoke`: strict EAM alloy interop smoke (`task_eam_alloy.yaml`, sync mode).
- `materials_parity_pack.py`: strict materials parity suite against versioned fixture
  (`examples/interop/materials_parity_suite_v1.json`) with per-case thresholds for
  force/energy/virial/temperature/pressure and trajectory/sync metrics.
- `paper`: broader sweep for reporting.
- `stress`: long + chaos.
- `async`: longer run without chaos, useful to study **asynchronous drift** metrics (`final_*`/`rms_*`).
- `longrun_envelope_ci`: long-run async sweep with strict baseline envelope gate from
  `golden/longrun_envelope_v1.json` (checks `final_*` and `rms_*` per row).
- `sync`: synchronous snapshot mode for long-run equivalence (serial vs TD-local).
- `paper_testcases_light`: shorter testcases sweep for quick local checks.
- `gpu_smoke`: strict short gate for GPU-track changes (see `docs/VERIFYLAB_GPU.md`).
- `gpu_interop_smoke`: strict task-interoperability gate for GPU-track changes.
- `gpu_metal_smoke`: strict EAM/materials task gate for GPU-track changes.
  - current implementation executes these presets with `device=cuda` and falls back to CPU if CUDA backend is unavailable.
- `gpu_smoke_hw`: current hardware-strict GPU gate (`require_effective_cuda`), fails on CUDA fallback.
- `gpu_interop_smoke_hw`: current hardware-strict interop GPU gate (`require_effective_cuda`), fails on CUDA fallback.
- `gpu_metal_smoke_hw`: current hardware-strict materials GPU gate (`require_effective_cuda`), fails on CUDA fallback.
- `mpi_overlap_smoke`: strict TD-MPI A/B overlap verification (`comm_overlap_isend` off/on) for ranks 2 and 4.
- `mpi_overlap_cudaaware_smoke`: strict TD-MPI A/B overlap verification with `cuda_aware_mpi=true` for overlap-on branch.
- `cluster_scale_smoke`: profile-driven strict strong/weak scaling gate (`scripts/bench_cluster_scale.py`).
- `cluster_stability_smoke`: profile-driven strict long-run stability gate (`scripts/bench_cluster_stability.py`).
- `mpi_transport_matrix_smoke`: profile-driven strict transport matrix gate (`scripts/bench_mpi_transport_matrix.py`).
- `metal_property_smoke`: strict property-level gate for metal cases (`case_prefix=eam_al_`) over `materials_parity_suite_v2`.
- `interop_metal_property_smoke`: strict property-level gate for alloy/interop cases (`case_prefix=eam_alloy_`) over `materials_parity_suite_v2`.
- `viz_smoke`: strict output/manifest/adapter contract gate for universal visualization.

Mode-level guarantees and gate ownership are summarized in `docs/MODE_CONTRACTS.md`.
Visualization governance and PR plan are defined in `docs/VISUALIZATION.md`.

## Strict Guardrails
- In `scripts/run_verifylab_matrix.py`, `--strict` enables guardrail hardening (`strict_guardrails=true`):
  - verification calls enforce `strict_min_zone_width=True` for TD-local checks;
  - invalid zone geometry (e.g. `zone width < cutoff`) is treated as error, not warning.
- Diagnostic mode (without `--strict`) keeps warning behavior.
- For task-based strict smoke presets, `zones_total_list` is set to geometry-valid values (`1`) to avoid false failures.

## Backend Evidence Artifacts
- `scripts/run_verifylab_matrix.py` stores backend evidence in:
  - `results/<run_id>/config.json` (`backend`, strict backend policy key),
  - `results/<run_id>/summary.json` (`backend`, `backend_ok`, strict backend policy key).
- `backend` payload includes:
  - `requested_device`,
  - `effective_device`,
  - `fallback_from_cuda`,
  - `reason`,
  - `warnings`.
- MPI overlap presets additionally persist per-rank overlap artifacts:
  - `results/<run_id>/mpi_overlap_n2.csv`,
  - `results/<run_id>/mpi_overlap_n2.md`,
  - `results/<run_id>/mpi_overlap_n4.csv`,
  - `results/<run_id>/mpi_overlap_n4.md`.

## CUDA-Cycle VerifyLab Plan (PR-C01..PR-C08)
- Keep all existing strict gates mandatory during CUDA kernel migration/hardening.
- Keep hardware-strict CUDA policy unchanged:
  - `gpu_smoke_hw`,
  - `gpu_interop_smoke_hw`,
  - `gpu_metal_smoke_hw`.
- Acceptance policy remains unchanged:
  - hardware-strict GPU lanes fail on any CPU fallback.

## Incident Bundle (R7)
- Any `--strict` failure in `scripts/run_verifylab_matrix.py` auto-creates:
  - `results/<run_id>/incident_bundle/manifest.json`,
  - `results/<run_id>/incident_bundle/README.txt`,
  - `results/<run_id>/incident_bundle/artifacts/*`.
- Bundle includes reproducibility artifacts when present:
  - `config.json`, `summary.json`, `summary.md`, `metrics.csv`,
  - overlap artifacts (`mpi_overlap_*.csv|*.md`),
  - trace/log files (`td_trace*.csv`, `*.log`, `*.txt`).
- `summary.json` and `summary.md` include `incident_bundle` metadata for failed strict runs.
- Envelope baseline can be overridden for diagnostics/tests:
  - `--envelope-file /path/to/envelope.json`.

## Task Runtime Contract (interop_smoke)
- `interop_smoke` uses task-based initialization and applies run-time compatibility checks from
  `tdmd/io/task.py::validate_task_for_run`.
- Ensemble input contract details are defined in `docs/ENSEMBLES.md`.
- Current run-time support:
  - `serial`: `nve/nvt/npt`,
  - `td_local`: `nve/nvt/npt`,
  - `td_full_mpi`: `nve/nvt` generally, `npt` with `MPI size=1`,
  - periodic boundaries in all directions (`box.pbc = [true,true,true]`),
  - cubic box (`x == y == z`),
  - potentials `lj` / `morse` / `table`,
  - pair runtime (`serial`/`td_local`) accepts multi-type and multi-mass tasks,
  - type-dependent pair coefficients for `lj`/`morse` via `potential.params.pair_coeffs` (complete matrix over task atom types),
  - table runtime via `potential.params.file` + `potential.params.keyword` (LAMMPS table subset parser).
  - strict potential params per potential kind.
- EAM/alloy runtime status:
  - `serial` supports `potential.kind=eam/alloy` (`eam_alloy` alias) with setfl tables and `elements` mapping.
  - `td_local` supports EAM backend; parity with serial is verified by EAM-specific tests (including `sync_mode=True` path).
  - `td_full_mpi` compute path supports EAM backend through automaton target-force callback (`forces_on_targets`), preserving TD state machine.
  - task-driven `td_full_mpi` accepts non-uniform per-atom masses and forwards mass arrays to the TD-MPI integrator path.
  - `scripts/run_verifylab_matrix.py --preset metal_smoke` and `--preset interop_metal_smoke` provide strict material smoke gates.
- Interop-only fields for now (parsed/exported but not executed in TDMD run):
  - legacy top-level `thermostat` (backward-compatible alias, non-executable),
  - charged atoms.

## Zone Width Guardrail
For 1D slab mode, `ZoneLayout1DCells` enforces `min_zone_width >= cutoff`:
- Default: emits a warning if any zone is narrower than `cutoff`.
- Strict mode: set `td.strict_min_zone_width: true` to raise an error instead.

## Async Example (cfg_system)
Run:
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset async --run-id async_local
```

Example summary (from `results/async_local/metrics.csv`, 2026‑02‑05):
- rows: `2` (zones_total = 4, 8)
- `final_dr` max: `3.179e+00` (zt=4), `7.802e+01` (zt=8)
- `final_dE` max: `1.423e+02` (zt=4), `2.634e+02` (zt=8)
- `rms_dr` max: `5.528e+01`
- `rms_dE` max: `1.331e+02`

Interpretation:
- `final_*` captures the drift at the **last observer step** (end-of-run).
- `rms_*` summarizes **typical drift** across the run.
- Increasing `zones_total` generally increases asynchrony and drift.

## Sync Example (cfg_system)
Run:
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset sync --run-id sync_local
```
Example summary (from `results/sync_local/metrics.csv`, 2026‑02‑05):
- rows: `2` (zones_total = 4, 8)
- `final_dr` max: `0.000e+00`
- `final_dv` max: `8.882e-16`
- `rms_dr` max: `0.000e+00`
- `rms_dv` max: `3.876e-16`

## Paper Testcases (gas + crystal)
Run (LJ config + testcases):
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_lj.yaml --preset paper --cases-mode testcases --golden write
python scripts/run_verifylab_matrix.py examples/td_1d_lj.yaml --preset paper --cases-mode testcases --golden check
```

Quick local check:
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_lj.yaml --preset paper_testcases_light --cases-mode testcases
```

## Strict mode vs regression mode
- Strict mode (`--strict`): exits non-zero if any row has `ok=false`.
- Regression mode (without `--strict`): records metrics/artifacts even if there are non-ok rows.
- Recommended:
  - CI gate: `smoke_ci --strict`.
  - Diagnostic trend tracking: `smoke_regression` (optionally without `--strict`).

## Long-run Envelope (R6)
- Baseline file: `golden/longrun_envelope_v1.json`.
- Gate owner: `scripts/run_verifylab_matrix.py` (`apply_envelope_gate`).
- Row identity for baseline matching:
  - `case`,
  - `zones_total`,
  - `use_verlet`,
  - `verlet_k_steps`,
  - `chaos_mode`,
  - `chaos_delay_prob`.
- Enforced metrics:
  - `final_dr`, `final_dv`, `final_dE`, `final_dT`, `final_dP`,
  - `rms_dr`, `rms_dv`, `rms_dE`, `rms_dT`, `rms_dP`.
- Artifacts:
  - `results/<run_id>/summary.json` includes `envelope` section with pass/fail, row counts, and violations.
  - `results/<run_id>/summary.md` includes envelope gate summary.

## Materials Parity Suite (R3)
- Fixture: `examples/interop/materials_parity_suite_v1.json` (`suite_version=1`).
- Scope:
  - `eam_al_serial_td_sync`,
  - `eam_alloy_serial_td_sync`.
- Checked properties per case:
  - initial: forces, PE, virial, T, P,
  - serial trajectory endpoint: positions, velocities, E/KE/PE/T/P,
  - serial-vs-td_local(sync) metrics: `max_*`, `final_*`, `rms_*`.
- Threshold policy (v1 fixture):
  - `force_abs`, `energy_abs`, `virial_abs`, `temp_abs`, `press_abs`,
  - `traj_r_abs`, `traj_v_abs`, `verify_abs`.
- Artifacts:
  - JSON summary is written to `results/materials_parity_suite_v1_summary.json` (or `--out` path).

## Materials Parity Suite v2 (R3/M1/M2/M3)
- Fixtures:
  - template: `examples/interop/materials_parity_suite_v2_template.json`,
  - generated reference: `examples/interop/materials_parity_suite_v2.json` (`suite_version=2`).
- Coverage extension:
  - Al and Al-Ni tasks at multiple temperature/density tags (`base`, `dense`).
- Generation tooling:
  - `python scripts/generate_materials_parity_fixture.py --template examples/interop/materials_parity_suite_v2_template.json --out examples/interop/materials_parity_suite_v2.json`
- Property-level reporting:
  - `scripts/materials_parity_pack.py` now emits `by_property` (`eos`, `thermo`, `transport`) and per-case `property_checks`.
- Dedicated property strict gate:
  - `scripts/materials_property_gate.py --fixture examples/interop/materials_parity_suite_v2.json --case-prefix eam_al_ --strict`
  - `scripts/materials_property_gate.py --fixture examples/interop/materials_parity_suite_v2.json --case-prefix eam_alloy_ --strict`
- Threshold policy calibration:
  - `python scripts/calibrate_material_thresholds.py --fixture examples/interop/materials_parity_suite_v2.json --out-json golden/material_threshold_policy_v2.json --out-md results/material_threshold_policy_v2.md`
