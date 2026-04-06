# VerifyLab

CUDA planning reference: `docs/CUDA_EXECUTION_PLAN.md` (PR-C01..PR-C08).

How to run:
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict
```
Long-run/operator telemetry during direct runtime runs:
```bash
python -m tdmd.main run --task examples/interop/task_eam_al.yaml --mode td_local --device cuda \
  --telemetry results/eam_run/telemetry.jsonl --telemetry-every 5 \
  --telemetry-heartbeat-sec 5 --telemetry-stdout
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
CUDA perf smoke (transfer/kernel latency gate):
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_perf_smoke --strict
```
EAM/alloy CPU-vs-GPU and time-vs-space decomposition benchmark:
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset eam_decomp_perf_smoke --strict
```

## CI Smoke
- CI/unit tests use the config-defined random system (`cfg_system`) for a **single-step** serial vs TD-local check.
- Strict CI gate:
  `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict`
- Standard PR observability benchmark:
  `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset eam_decomp_perf_smoke --strict`
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
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset mpi_overlap_async_observe_smoke --strict
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset mpi_overlap_cudaaware_async_observe_smoke --strict
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset mpi_overlap_perf_observe_smoke --strict
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
- `gpu_perf_smoke`: hardware-strict CUDA perf smoke (`require_effective_cuda`) reporting
  `h2d_full_ms`, `h2d_delta_ms`, `kernel_ms`, `d2h_ms`, `delta_over_full`, `transfer_over_kernel`.
  The synthetic kernel is calibrated to a minimum timing floor before thresholding so the
  micro-perf guardrail is less sensitive to sub-0.1 ms timer noise.
- `ml_reference_smoke`: strict CPU task-based VerifyLab lane for the `quadratic_density`
  `ml/reference` family. It uses `examples/interop/task_ml_reference.yaml` in `sync_mode=True` to
  enforce serial-vs-`td_local` parity on the CPU reference harness.
- `ml_reference_parity_smoke`: strict external fixture lane for the same `quadratic_density`
  family. It wraps `scripts/ml_reference_parity_pack.py` over
  `examples/interop/ml_reference_suite_v1.json`.
- `eam_decomp_perf_smoke`: standard PR benchmark for `EAM/alloy` that emits a four-column table
  (`space_cpu`, `space_gpu`, `time_cpu`, `time_gpu`) plus derived speedup rows for
  GPU-vs-CPU and time-vs-space comparisons. This is an execution/observability benchmark, not a hard performance-threshold gate.
- `eam_decomp_perf_gpu_heavy`: manual GPU-only `EAM/eam-alloy` benchmark for operator-side perf
  evaluation. It runs a `10K`-atom model with heavier `steps/zones`, compares `space_gpu` vs
  `time_gpu`, requires effective CUDA, and is intended to stay roughly within a single
  multi-minute run budget rather than PR CI timing.
- `eam_decomp_zone_sweep_gpu`: manual GPU-only zone sweep for `10K`-atom `EAM/eam-alloy`.
  It compares `space_gpu` vs `time_gpu` across several valid zone layouts and emits a best-observed
  layout summary. This is observability-only and is not a PR gate.
- `eam_td_breakdown_gpu`: manual GPU-only `10K`-atom `EAM/eam-alloy` breakdown for the current
  best observed `2-zone` layout. It compares `space_gpu` vs `time_gpu` and attributes runtime to
  `forces_full`, `target_local_force`, nested device sync, cell-list build, candidate enumeration,
  and zone bookkeeping. The artifact also records the current many-body force-scope contract
  (`evaluation_scope`, `consumption_scope`, `target_local_available`) plus
  `baseline_reference_version=pr_mb01_v1`, so current runs can be measured against the frozen
  pre-locality ceiling. After `PR-MB03`, current GPU runs should report `target_local` scope with
  reduced or eliminated `forces_full` share relative to that baseline. Use it to distinguish TD
  algorithmic headroom from backend/runtime overhead before changing zoning policy.
- `td_autozoning_advisor_gpu`: manual resource-aware GPU-only TD zoning advisor for `EAM/eam-alloy`.
  It detects visible CPU/GPU/MPI resources, enumerates strict-valid layout candidates, benchmarks
  paired `space_gpu` vs `time_gpu` rows, and emits a recommendation-only zoning plan. When
  breakdown is enabled, it also attaches `pr_za01_v1` force-scope evidence for the recommended
  layout using `baseline_reference_version=pr_mb03_v1`. This is observability/operator guidance,
  not a runtime policy switch.
- `al_crack_100k_compare_gpu`: manual operator benchmark for a pure-Al `100K` microcrack task with
  `eam/alloy`, requested `1000` zones, and GPU-only execution. It emits an exact-request
  `space_gpu_z1000` row, explicit TD preflight evidence for the requested `1000`-zone time
  decomposition, and, when needed, a strict-valid common-zone fallback comparison with per-case
  telemetry sidecars. Use it for realistic long-run GPU/operator comparison rather than PR CI.
- `scripts/profile_gpu_backend.py`: consolidated CUDA-cycle profiling helper for operator use.
  It combines verify-preset timing ratios, `gpu_perf_smoke`, `EAM/alloy` decomposition benchmarking,
  and optional `Phase E` comparison into one markdown/json report.
- `mpi_overlap_smoke`: strict TD-MPI A/B overlap verification (`comm_overlap_isend` off/on) for ranks 2 and 4.
- `mpi_overlap_cudaaware_smoke`: strict TD-MPI A/B overlap verification with `cuda_aware_mpi=true` for overlap-on branch, plus CUDA-aware activity guard (`cuda_aware_active_ok=1`, no CPU fallback rank lines).
- `mpi_overlap_async_observe_smoke`: strict TD-MPI A/B overlap verification with explicit async evidence gate (`async_send_msgs_max`/`async_send_bytes_max` > 0 for `overlap=1` rows).
- `mpi_overlap_cudaaware_async_observe_smoke`: same async evidence gate, with `cuda_aware_mpi=true` on overlap-on branch and CUDA-aware activity guard.
- `mpi_overlap_perf_observe_smoke`: async evidence gate + overlap window evidence (`overlap_window_ms_max > 0` for `overlap=1` rows), plus timing observability metrics (`send_pack_ms_max`, `send_wait_ms_max`, `recv_poll_ms_max`).
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
- `eam_decomp_perf_smoke` additionally persists:
  - `results/<run_id>/eam_decomp_perf.csv`,
  - `results/<run_id>/eam_decomp_perf.md`,
  - `results/<run_id>/eam_decomp_perf.summary.json`,
  and prints the benchmark markdown table to stdout after the run.
- `eam_decomp_perf_gpu_heavy` additionally persists:
  - `results/<run_id>/eam_decomp_perf_gpu_heavy.csv`,
  - `results/<run_id>/eam_decomp_perf_gpu_heavy.md`,
  - `results/<run_id>/eam_decomp_perf_gpu_heavy.summary.json`.
- `eam_decomp_zone_sweep_gpu` additionally persists:
  - `results/<run_id>/eam_decomp_zone_sweep_gpu.csv`,
  - `results/<run_id>/eam_decomp_zone_sweep_gpu.md`,
  - `results/<run_id>/eam_decomp_zone_sweep_gpu.summary.json`.
- `eam_td_breakdown_gpu` additionally persists:
  - `results/<run_id>/eam_td_breakdown_gpu.csv`,
  - `results/<run_id>/eam_td_breakdown_gpu.md`,
  - `results/<run_id>/eam_td_breakdown_gpu.summary.json`.
- `td_autozoning_advisor_gpu` additionally persists:
  - `results/<run_id>/td_autozoning_advisor_gpu.csv`,
  - `results/<run_id>/td_autozoning_advisor_gpu.md`,
  - `results/<run_id>/td_autozoning_advisor_gpu.summary.json`.
- `ml_reference_parity_smoke` additionally persists:
  - `results/<run_id>/ml_reference_parity.summary.json`.
- `scripts/profile_gpu_backend.py` additionally persists:
  - `results/gpu_profile.csv`,
  - `results/gpu_profile.md`,
  - `results/gpu_profile.summary.json`,
  plus nested artifacts for `gpu_perf_smoke` and current/optional-Phase-E `EAM/alloy` benchmark runs.
- Direct `tdmd.main run` executions may additionally persist:
  - `<telemetry>.manifest.json`,
  - `<telemetry>.summary.json`,
  when `--telemetry` is enabled.
- MPI overlap presets additionally persist per-rank overlap artifacts:
  - `results/<run_id>/mpi_overlap_n2.csv`,
  - `results/<run_id>/mpi_overlap_n2.md`,
  - `results/<run_id>/mpi_overlap_n4.csv`,
  - `results/<run_id>/mpi_overlap_n4.md`.
- overlap CSV rows include CUDA-aware activity observability for strict transport guards:
  - `backend_cuda_lines`,
  - `backend_cpu_lines`,
  - `cuda_aware_active_ok`.

## CUDA-Cycle VerifyLab Plan (PR-C01..PR-C08)
- Keep all existing strict gates mandatory during CUDA kernel migration/hardening.
- Keep hardware-strict CUDA policy unchanged:
  - `gpu_smoke_hw`,
  - `gpu_interop_smoke_hw`,
  - `gpu_metal_smoke_hw`.
- Acceptance policy remains unchanged:
  - hardware-strict GPU lanes fail on any CPU fallback.
- Current operator guidance:
  use `scripts/profile_gpu_backend.py` with representative `EAM/alloy` workload
  (`--eam-n-atoms 512 --eam-steps 2`) before reconsidering the current `stay_on_rawkernel`
  stack decision.

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
- ML reference runtime status (`PR-ML01`):
  - `potential.kind=ml/reference` is a CPU-reference many-body harness with versioned
    `contract.cutoff`, `contract.descriptor`, `contract.neighbor`, and `contract.inference`.
  - `serial`, `td_local`, task-driven CLI runtime, and task-based VerifyLab sweep path accept it.
  - runtime compatibility requires `task.cutoff >= contract.cutoff.radius`.
  - `contract.neighbor.requires_full_system_barrier` must stay `false`; hidden global barriers are
    rejected at task validation time.
  - `PR-ML02` adds strict acceptance for the `quadratic_density` family via
    `ml_reference_smoke` plus versioned fixture suite
    `examples/interop/ml_reference_suite_v1.json` checked by
    `scripts/ml_reference_parity_pack.py --strict`.
- Interop-only fields for now (parsed/exported but not executed in TDMD run):
  - legacy top-level `thermostat` (backward-compatible alias, non-executable),
  - charged atoms.
  - `ml/reference` LAMMPS export (explicitly rejected by `tdmd/io/lammps.py` today).

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
