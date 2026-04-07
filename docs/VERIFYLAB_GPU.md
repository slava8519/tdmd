# VerifyLab GPU

CUDA-cycle planning reference: `docs/CUDA_EXECUTION_PLAN.md` (PR-C01..PR-C08).

## Purpose
GPU verification keeps CPU as reference and checks numerical equivalence under strict tolerances.
Cross-mode guarantees and gate ownership are documented in `docs/MODE_CONTRACTS.md`.

## Active Presets
- `gpu_smoke`
  - command:
    ```bash
    python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_smoke --strict
    ```
  - intent: short strict parity gate for GPU-track changes.
- `gpu_interop_smoke`
  - command:
    ```bash
    python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_interop_smoke --strict
    ```
  - intent: strict interop parity gate on task-based initialization.
- `gpu_metal_smoke`
  - command:
    ```bash
    python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_metal_smoke --strict
    ```
  - intent: strict materials/EAM task parity gate for GPU-track changes.
- Hardware-strict variants (fail on CUDA->CPU fallback):
  - `gpu_smoke_hw`
  - `gpu_interop_smoke_hw`
  - `gpu_metal_smoke_hw`
  - `gpu_perf_smoke`
  - command pattern:
    ```bash
    python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_smoke_hw --strict
    ```
  - perf smoke command:
    ```bash
    python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_perf_smoke --strict
    ```

## Historical and CUDA-Cycle Notes
- Historical `PR-G01` introduced the initial strict verification gate shape for the GPU track.
- CUDA-cycle operation is now governed by `docs/CUDA_EXECUTION_PLAN.md` and the consolidated
  `scripts/profile_gpu_backend.py` workflow rather than open-ended future TODO items.
- In `PR-G02`, pairwise `LJ/Morse` force kernels can run through CUDA backend for `serial`/`td_local`; GPU presets execute with `device=cuda` (with CPU fallback when CUDA is unavailable).
- In `PR-G03`, GPU path can use cell-list candidate pruning (`forces_on_targets_celllist_backend`) to keep candidate-set parity with CPU cell-list behavior.
- In `PR-G04`, `table` potential is included in GPU force runtime path (`serial`/`td_local` dispatch).
- In `PR-G05`, EAM/alloy force path is included in GPU backend dispatch.
- In `PR-C04`, direct GPU target-force evaluation for `table` and `eam/alloy` no longer uses the
  dense fallback path:
  - `table` direct path runs through the neighbor-list RawKernel force pipeline,
  - `eam/alloy` direct path uses device-side two-pass kernels over the candidate-union neighbor list.
- In `PR-C05`, runtime-managed CUDA paths in `serial` and `td_local` keep device positions/types hot
  across timesteps and use explicit dirty-range syncing after host-side position updates. Static
  `table` and `eam/alloy` potential tables are cached on device to avoid repeated uploads.
- In `PR-C06`, the TD-MPI CUDA refinement path opts into the same dirty-range runtime policy:
  work-zone GPU force calls use `prefer_marked_dirty=True`, MPI receive updates mark only the
  received atom ids dirty, and NPT box rescale marks device state dirty only when positions change.
- In `PR-G06`, td_local host-automaton pipeline can execute device-backed force computations while preserving TD state semantics.
- In `PR-G07`, multi-rank launch path uses local-rank GPU mapping (`1 rank = 1 GPU` policy) for TD-MPI execution.
- In `PR-G08`, TD-MPI transport supports optional overlap mode (`td.comm_overlap_isend`) and optional CUDA-aware transport toggle (`td.cuda_aware_mpi`) with host-staging fallback when CUDA-aware path is inactive.
- In post-baseline risk burndown `PR-H01`, VerifyLab GPU presets support hardware-strict no-fallback gating:
  - `scripts/run_verifylab_matrix.py` stores backend evidence in `config.json`/`summary.json`,
  - `require_effective_cuda` fails strict run when requested `cuda` resolves to CPU fallback.
- In `PR-H03`, strict VerifyLab mode enables geometry guardrails (`strict_min_zone_width=True`) for TD-local verification calls.
- In `PR-H06`, long-run async drift governance is enforced via `longrun_envelope_ci` baseline gate.
- In `PR-H07`, strict failure runs auto-generate incident bundles for reproducible triage.

MPI smoke helpers:
```bash
python scripts/run_mpi_smoke.py --n 2 --config examples/td_1d_morse_static_rr.yaml
python scripts/run_mpi_smoke.py --n 4 --config examples/td_1d_morse_static_rr_smoke4.yaml
```

MPI overlap benchmark helper:
```bash
python scripts/bench_mpi_overlap.py --config examples/td_1d_morse_static_rr_smoke4.yaml --n 4 --overlap-list 0,1 --cuda-aware --out results/mpi_overlap.csv --md results/mpi_overlap.md
```

MPI overlap strict presets in VerifyLab matrix:
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset mpi_overlap_smoke --strict
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset mpi_overlap_cudaaware_smoke --strict
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset mpi_overlap_async_observe_smoke --strict
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset mpi_overlap_cudaaware_async_observe_smoke --strict
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset mpi_overlap_perf_observe_smoke --strict
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset mpi_transport_matrix_smoke --strict
```

MPI overlap artifacts also include async transport observability counters:
- `async_send_msgs_max`
- `async_send_bytes_max`
- `async_evidence_ok` (strict evidence status per row for overlap lanes when enabled)
- PR-C06 keeps these overlap diagnostics active while integrating runtime-managed CUDA dirty-range sync
  into `td_full_mpi`; overlap/cluster/transport strict lanes remain the acceptance evidence.
- CUDA-aware activity guard counters:
  - `backend_cuda_lines`
  - `backend_cpu_lines`
  - `cuda_aware_active_ok` (strict status when CUDA-aware activity is required)
- timing counters:
  - `send_pack_ms_max`
  - `send_wait_ms_max`
  - `recv_poll_ms_max`
  - `overlap_window_ms_max`
  - `overlap_window_ok` (strict window evidence status for overlap lanes when enabled)

Cluster-profile transport matrix helper (fabric-aware, profile-driven):
```bash
python scripts/bench_mpi_transport_matrix.py --profile examples/cluster/cluster_profile_real_template.yaml --strict
```

## Profiling Helper
Generate comparative CPU-vs-GPU-track timing and parity report:
```bash
python scripts/profile_gpu_backend.py --config examples/td_1d_morse.yaml --out-csv results/gpu_profile.csv --out-md results/gpu_profile.md --out-json results/gpu_profile.summary.json --require-effective-cuda --eam-n-atoms 512 --eam-steps 2
```

Optional Phase E comparison:
```bash
git worktree add /tmp/tdmd_phase_e efb864e
python scripts/profile_gpu_backend.py --config examples/td_1d_morse.yaml --out-csv results/gpu_profile.csv --out-md results/gpu_profile.md --out-json results/gpu_profile.summary.json --require-effective-cuda --eam-n-atoms 512 --eam-steps 2 --phase-e-worktree /tmp/tdmd_phase_e
```

The consolidated profiler keeps the legacy preset timing CSV and adds:
- `gpu_perf_smoke` micro-perf metrics,
- current `EAM/alloy` decomposition benchmark artifacts,
- optional `Phase E` speedup comparison,
- a `Plan B Decision` section in markdown/json output.
- `gpu_perf_smoke` uses a calibrated synthetic kernel timing floor before applying the
  `transfer_over_kernel` envelope so the micro-perf signal is less sensitive to timer jitter.
- For heavier operator-side CUDA benchmarking, `eam_decomp_perf_gpu_heavy` runs a GPU-only
  `10K`-atom `EAM/eam-alloy` comparison between `space_gpu` and `time_gpu`.
- `eam_decomp_zone_sweep_gpu` extends that benchmark into a manual zone-layout sweep so operators
  can see which valid TD/space zone counts are favorable before any future automation.
  Current summaries also expose passive `PR-SW01` wavefront contract fields per layout.
  `PR-SW05` additionally exports realized runtime wave-batch cost fields from `pr_sw05_v1`
  (launches saved, neighbor reuse, and candidate-union pressure) for the `time_gpu` rows.
- `eam_td_breakdown_gpu` profiles the current best observed `10K`-atom layout and splits the GPU
  runtime into `forces_full`, `target_local_force`, nested device sync, cell-list build,
  candidate enumeration, and zone bookkeeping. It also records the current many-body force-scope
  contract (`evaluation_scope`, `consumption_scope`, `target_local_available`) plus
  `baseline_reference_version=pr_mb01_v1`, so the large-run baseline stays stable while current
  CUDA runs can be compared against the frozen pre-locality ceiling. After `PR-MB03`, current GPU
  runs should report `target_local` scope and reduced `forces_full` share. Use it when TD speedup
  looks smaller than theory suggests and you need to separate backend overhead from true TD
  headroom. `PR-SW05` adds aggregated `wave_batch_diagnostics` so the same artifact now reports
  realized occupancy/launch-reduction/reuse signals for CUDA `1D` slab batching.
- `td_autozoning_advisor_gpu` turns that corrected locality evidence into a recommendation-only
  zoning report. It detects visible CPU/GPU/MPI resources, benchmarks strict-valid candidate
  layouts, and emits markdown/json/csv artifacts with a recommended TD layout and optional
  `pr_za01_v1` breakdown evidence. Current summaries also expose passive `PR-SW01` wavefront
  contract fields per layout and for the recommendation. `PR-SW05` additionally feeds runtime
  `pr_sw05_v1` wave-batch cost fields into the per-layout and recommendation artifacts.
  It does not auto-apply runtime policy.
- `al_crack_100k_compare_gpu` is the first-class operator benchmark for a pure-Al `100K`
  microcrack on GPU. It preserves an exact-request `space_gpu_z1000` row, records whether the
  requested `time_gpu_z1000` TD layout is geometrically valid, falls back to a strict-valid common
  zone count when needed, and persists per-case telemetry sidecars so timeout runs still report the
  last observed step, RSS, and CUDA memory. Current summaries also include exact-request and
  strict-valid passive `PR-SW01` wavefront preflight blocks.
- `slab_wavefront_evidence_gpu` is the first-class operator evidence pack for `PR-SW02`.
  It keeps `al_crack_100k_compare_gpu`, adds the crack `z=1..12` sweep, reuses an
  `EAM/eam-alloy` control sweep, and attaches one control breakdown so the output distinguishes:
  - TD vs space at equal `z`,
  - TD absolute runtime vs `z`,
  - orchestration vs force-kernel share.
  `PR-SW05` extends the same evidence pack with runtime wave-batch cost views for the control
  sweep/breakdown (`launches_saved_per_step`, neighbor reuse, candidate-union pressure).
  The exact-crack section is accepted on evidence completeness, not on raw child-benchmark
  `ok_all`: the over-requested `space_gpu_z1000` case may still time out as long as the
  requested-layout preflight and strict-valid common-zone comparison are both recorded.
  Equal-`z` crack-sweep `space` controls now use a `420s` budget as well; the earlier `90s`
  setting truncated valid `100K` `space` runs from `z≈4` upward and was treated as an operator
  budget issue rather than a wavefront verdict.
  For the small `10K` alloy control case, the preset now keeps only common-valid `z=1..8` for
  `space` vs `1D` TD comparison and uses `z=8` for the control breakdown.
  The small `10K` control sweep now records `skipped_invalid_space_zone_totals` for requested
  `zones_total` values that have no strict-valid `3D` `space` layout instead of aborting the
  evidence pack.
- `wavefront_reference_smoke` is the strict CPU/reference proof gate for `PR-SW03`.
  It remains CPU-only, proves grouped admissible slab waves against the current sequential CPU
  slab semantics, and freezes the many-body baseline as current target-local
  `sequential_1d_many_body_target_local` before `PR-SW04` introduces any CUDA batching path.
- `PR-SW04` now provides the CUDA runtime contract `pr_sw04_v1`:
  admissible `1D` slab waves fuse the pre-force half-step on CUDA while zone state progression
  remains sequential-per-zone. Existing `gpu_*` strict lanes cover this runtime path; the
  operator-side performance decision still belongs to `PR-SW05`, which adds the passive runtime
  diagnostics contract `pr_sw05_v1`.
- Current decision policy:
  stay on `CuPy RawKernel` unless representative profiling stops beating both CPU reference and
  the archived `Phase E` baseline.

Transfer/kernel perf smoke helper:
```bash
python scripts/bench_gpu_perf_smoke.py --out-json results/gpu_perf_smoke.summary.json --strict
```

## Archived Portability Notes
- These notes are retained only as historical background for the pre-CUDA portability discussion.
- They are not the active acceptance policy for the completed CUDA cycle.
- Migration did not weaken current strict gates; existing `gpu_*` presets remain mandatory.
- Historical planned strict-gate evolution:
  - backend-agnostic key `require_effective_gpu` replaces CUDA-specific policy key (with compatibility alias),
  - vendor-specific hardware lanes:
    - `gpu_cuda_smoke_hw`,
    - `gpu_hip_smoke_hw`,
  - cross-vendor parity lane:
    - `gpu_portability_smoke`.
- Hardware-strict rule is unchanged:
  - CPU fallback is failure when a hardware GPU backend is requested.
