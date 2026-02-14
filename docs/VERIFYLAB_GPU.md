# VerifyLab GPU

Portability-cycle planning reference: `docs/PORTABILITY_KOKKOS_PLAN.md` (PR-K01..PR-K10).

## Purpose
GPU verification keeps CPU as reference and checks numerical equivalence under strict tolerances.
Cross-mode guarantees and gate ownership are documented in `docs/MODE_CONTRACTS.md`.

## Current Preset
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

## Stage Notes
- In current stage (`PR-G01`) this preset validates the strict verification gate shape for the GPU track.
- Dedicated kernel-specific parity thresholds and hardware CI matrix expansion are planned for later TODO items.
- In `PR-G02`, pairwise `LJ/Morse` force kernels can run through CUDA backend for `serial`/`td_local`; GPU presets execute with `device=cuda` (with CPU fallback when CUDA is unavailable).
- In `PR-G03`, GPU path can use cell-list candidate pruning (`forces_on_targets_celllist_backend`) to keep candidate-set parity with CPU cell-list behavior.
- In `PR-G04`, `table` potential is included in GPU force runtime path (`serial`/`td_local` dispatch).
- In `PR-G05`, EAM/alloy force path is included in GPU backend dispatch.
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
python scripts/profile_gpu_backend.py --config examples/td_1d_morse.yaml --out-csv results/gpu_profile.csv --out-md results/gpu_profile.md
```

Transfer/kernel perf smoke helper:
```bash
python scripts/bench_gpu_perf_smoke.py --out-json results/gpu_perf_smoke.summary.json --strict
```

## Portability-Cycle Gate Plan
- Migration does not weaken current strict gates; existing `gpu_*` presets remain mandatory.
- Planned strict-gate evolution:
  - backend-agnostic key `require_effective_gpu` replaces CUDA-specific policy key (with compatibility alias),
  - vendor-specific hardware lanes:
    - `gpu_cuda_smoke_hw`,
    - `gpu_hip_smoke_hw`,
  - cross-vendor parity lane:
    - `gpu_portability_smoke`.
- Hardware-strict rule is unchanged:
  - CPU fallback is failure when a hardware GPU backend is requested.
