# GPU Backend

Active CUDA execution cycle: `docs/CUDA_EXECUTION_PLAN.md` (PR-C01..PR-C08, CuPy RawKernel).
Historical portability reference (archived): `docs/PORTABILITY_KOKKOS_PLAN.md`.

## Scope
GPU support is implemented as a refinement path over the CPU reference semantics.
The TD automaton (F/D/P/W/S transitions), dependency predicates, and invariants remain unchanged.
Mode-level guarantees and strict-gate ownership are documented in `docs/MODE_CONTRACTS.md`.

## Current Stage (PR-G01)
- Added backend selection abstraction in `tdmd/backend.py`.
- Supported device selectors:
  - `auto`
  - `cpu`
  - `cuda`
- Added config field `run.device` (`auto|cpu|cuda`) and CLI override `tdmd run --device ...`.
- `cuda` selection is capability-aware:
  - if CUDA/CuPy is available, backend resolves to `cuda`;
  - otherwise runtime warns and falls back to `cpu`.
- `TDMD_FORCE_CPU=1` forces CPU when `run.device=auto`.

## Active CUDA Cycle (CuPy RawKernel)
- Migrating from CuPy high-level API to CuPy RawKernel (real CUDA C kernels).
- Key improvements: O(N) neighbor lists, single-launch force kernels, fused EAM,
  persistent GPU state, CUDA stream overlap.
- Stack: CuPy RawKernel primary; C++/CUDA extension as Plan B.
- Numba-CUDA is not in scope.
- Full plan: `docs/CUDA_EXECUTION_PLAN.md`.

## Semantics Guarantee
- CPU path remains the formal reference.
- Backend selection does not alter TD scheduling semantics.
- No global barrier is introduced by backend selection.

## Notes
- At this stage backend selection is wired and validated; full GPU kernels and multi-GPU execution are tracked in TODO Phase E (`PR-G02+`).
- Profiling helper: `scripts/profile_gpu_backend.py` writes CPU-vs-GPU-track timing/parity artifacts (`results/gpu_profile.csv`, `results/gpu_profile.md`).

## Current Stage (PR-G02)
- Added CUDA pair-force kernels for `LJ/Morse` in `tdmd/forces_gpu.py`.
- `tdmd/serial.py` and `tdmd/td_local.py` dispatch to GPU pair kernels when:
  - `device` resolves to `cuda`,
  - potential is pairwise (`LJ/Morse`).
- If CUDA backend is unavailable or potential is unsupported, code paths fall back to CPU force kernels.

## Current Stage (PR-G03)
- Added GPU cell-list candidate path in `tdmd/forces_gpu.py::forces_on_targets_celllist_backend`.
- Candidate sets are built with CPU-equivalent cell buckets and neighbor-stencil logic, then pair interactions are evaluated on GPU.
- Parity checks are covered by tests (`tests/test_gpu_pair_kernels.py`) with CUDA-enabled assertions when hardware/backend is available.

## Current Stage (PR-G04)
- Extended GPU force runtime to `table` potential in `tdmd/forces_gpu.py`.
- `serial` and `td_local` can execute table force evaluations through the same backend-dispatch path used for pair kernels.

## Current Stage (PR-G05)
- Added EAM/alloy CUDA force-evaluation path in `tdmd/forces_gpu.py` (`EAMAlloyPotential` branch).
- `serial` and `td_local` can dispatch EAM force evaluation to GPU backend (with CPU fallback when CUDA is unavailable).

## Current Stage (PR-G06)
- `td_local` host automaton pipeline is wired to use GPU force backend for supported potentials while keeping existing TD scheduling/state logic unchanged.
- Sync and async td_local parity checks are covered in `tests/test_gpu_pair_kernels.py` (CUDA-enabled checks where backend is available).

## Current Stage (PR-G07)
- Added local-rank to device mapping helpers:
  - `tdmd/backend.py::local_rank_from_env`
  - `tdmd/backend.py::cuda_device_for_local_rank`
- `tdmd/td_full_mpi.py` uses rank-local device mapping for `1 rank = 1 GPU` assignment when CUDA backend is selected.
- MPI smoke coverage includes 2-rank and 4-rank static_rr cases (`examples/td_1d_morse_static_rr.yaml`, `examples/td_1d_morse_static_rr_smoke4.yaml`).

## Current Stage (PR-G08)
- Added optional MPI transport refinements in `tdmd/td_full_mpi.py`:
  - `td.comm_overlap_isend`: nonblocking `Isend` path for payload delivery with batched `Waitall`.
  - `td.cuda_aware_mpi`: enables CUDA-aware transport mode when backend resolves to `cuda` (falls back to host-staging when inactive).
- Added overlap benchmarking helper:
  - `scripts/bench_mpi_overlap.py` compares blocking vs overlap path and writes
    `results/mpi_overlap.csv` and `results/mpi_overlap.md`.
- These options are transport-level refinements and do not alter TD automaton state transitions.

## Post-Baseline Hardening (PR-H01)
- Added hardware-strict GPU verification gate in `scripts/run_verifylab_matrix.py`:
  - backend evidence is persisted to `results/<run_id>/config.json` and `results/<run_id>/summary.json`,
  - when `require_effective_cuda` is enabled, CUDA request resolving to CPU fallback is treated as verification failure.
- Added hardware-strict GPU presets:
  - `gpu_smoke_hw`, `gpu_interop_smoke_hw`, `gpu_metal_smoke_hw`.
- This changes verification policy only and does not alter runtime TD semantics.

## Strict-Gate Plan (CUDA Cycle)
- Existing strict GPU gates remain mandatory throughout migration:
  - `gpu_smoke`, `gpu_interop_smoke`, `gpu_metal_smoke`,
  - `gpu_smoke_hw` (CUDA hardware-strict lane),
  - `gpu_perf_smoke` (CUDA transfer/kernel perf smoke lane).
- No-fallback acceptance rule remains mandatory for hardware-strict lanes.

## Post-Baseline Hardening (PR-H02)
- Added strict TD-MPI overlap A/B presets in `scripts/run_verifylab_matrix.py`:
  - `mpi_overlap_smoke`,
  - `mpi_overlap_cudaaware_smoke`.
- Presets invoke `scripts/bench_mpi_overlap.py` for ranks `2` and `4`, persist per-rank artifacts, and enforce strict invariant counters (`hG`, `hV`, `violW`, `lagV`).
- This is verification/transport-policy hardening and does not alter TD automaton semantics.

## Post-Baseline Hardening (PR-H03)
- Strict VerifyLab runs now enforce geometry guardrails (`strict_min_zone_width=True`) in TD-local verification calls.
- Task-based strict smoke presets use geometry-valid zone layouts to avoid warning-only acceptance.
- This is acceptance-policy hardening only; compute/automaton semantics are unchanged.

## Post-Baseline Hardening (PR-H06/H07)
- Long-horizon async drift governance is enforced by `longrun_envelope_ci` and baseline file `golden/longrun_envelope_v1.json`.
- Strict VerifyLab failures auto-produce incident bundles (`tdmd/incident_bundle.py`) and can be manually exported via `scripts/export_incident_bundle.py`.
- These are verification/triage policy layers only; runtime TD semantics and backend execution semantics are unchanged.

## Risk Burndown v2 (PR-V2-01..PR-V2-04)
- Added profile-driven cluster validation contract (`tdmd/cluster_profile.py`, `examples/cluster/*.yaml`).
- Added strict transport matrix benchmarking (`scripts/bench_mpi_transport_matrix.py`) and VerifyLab preset `mpi_transport_matrix_smoke`.
- Added strict cluster scaling/stability lanes:
  - `cluster_scale_smoke`,
  - `cluster_stability_smoke`.
- These lanes validate overlap/cuda-aware transport behavior under profile-controlled conditions and do not alter TD automaton or backend semantics.
