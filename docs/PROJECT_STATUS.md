# Project status (post-CUDA maintenance cycle)

TDMD-TD implements **Time Decomposition Molecular Dynamics** with a strict TD automaton.

## Stable
- 1-zone formal core (`W<=1`), batched comm as refinement.
- Liveness via A4b total-order tie-break.
- verify v2 + invariants + chaos schedule.
- WFG local diagnostics + contention metrics (`wfgC_rate`, `wfgC_per_100_steps`).
- Pareto analysis in verifylab summary (speedup vs contention).

## Code Quality & Tooling
- **Pre-commit chain**: ruff (import sort + pyflakes F) → black (100 char) → mypy (typed island).
- **Mypy typed island** (9 modules): `atoms`, `output`, `observer`, `force_dispatch`,
  `cli_parser`, `constants`, `zones`, `backend`, `td_automaton`.
- **Named constants** (`tdmd/constants.py`): `NUMERICAL_ZERO`, `FLOAT_EQ_ATOL`, `GEOM_EPSILON`
  replace 30+ magic numbers across force kernels, geometry checks, and float comparisons.
- **Exception handling**: broad catches are minimized; optional-backend detection paths
  intentionally keep broad fallback handling to preserve CPU-safe execution.
- **`run_td_local` restructure**: 700-line god-function split into `_TDLocalCtx` dataclass +
  4 execution paths (`_run_sync_global`, `_run_sync_1d_zones`, `_run_async_3d`, `_run_async_1d`).
- **`run_td_full_mpi_1d` restructure**: 1400-line god-function decomposed in two passes:
  *Pass 1*: `_TDMPIRuntimeInit` + `_TDMPICommContext` dataclasses, extracted `_init_td_full_mpi_runtime`,
  `_build_zones_and_automaton`, `_recv_phase`, `_send_phase`, `_handle_record` (and sub-handlers),
  `_run_td_warmup_phase`, `_run_td_main_phase`.
  *Pass 2*: 17 closures (614→344 lines) replaced by 15 top-level helpers + `_TDMPISimState`
  mutable dataclass (`box`, `skin_global`); 43-line config-unpacking block eliminated via
  direct `config.X` access; thin closure adapters bind helpers to local state.
- **Config objects** (`tdmd/run_configs.py`): `TDLocalRunConfig`, `TDFullMPIRunConfig` with
  `from_legacy_kwargs()` for backward compatibility; replaces 50+ loose keyword arguments.
- **Test suite**: 227 tests passing, 3 skipped in the current post-CUDA maintenance worktree.

## GPU Backend Status
- **Phase E (complete, historical)**: CuPy high-level API path — functionally correct but
  performance-limited by O(N²) broadcast matrices, Python loops, and lack of persistent GPU state.
- **Phase H / CUDA execution cycle (complete)**: `PR-C01..PR-C08` implemented on **CuPy RawKernel**
  with O(N) neighbor lists, fused EAM kernels, persistent device state, TD-MPI CUDA integration,
  and profiling/operator guidance.
- **Current stack decision**: stay on **CuPy RawKernel**.
  Representative `gpu-profile` evidence shows current CUDA path beating both CPU reference and
  the archived `Phase E` baseline commit `efb864e`.
- **Plan B status**: `C++/CUDA` extension remains optional and should be reconsidered only after a
  fresh representative profiler run.

## Next
- Maintain the completed CUDA cycle under the current `CuPy RawKernel` policy.
- Re-run `scripts/profile_gpu_backend.py` before any future Plan B decision.
- Completed post-CUDA follow-ups:
  - `PR-MB01..PR-MB03` many-body locality hardening,
  - `PR-ZA01` recommendation-only auto-zoning advisor,
  - `PR-ML01..PR-ML02` CPU-reference ML contract/fixture groundwork.
- New active algorithmic priority:
  **single-GPU `1D` slab wavefront TD scaling** for large metals/alloys workloads.
- Completed first step:
  - `PR-SW01` formalized the passive slab-wavefront contract (`pr_sw01_v1`) and surfaced
    first-wave/deferred-zone/fallback observability in the current operator benchmarks.
- Completed second step:
  - `PR-SW02` added the first-class operator evidence pack (`slab_wavefront_evidence_gpu`) that
    preserves `al_crack_100k_compare_gpu`, adds a crack `z=1..12` sweep, and pairs it with an
    `EAM/eam-alloy` control sweep plus breakdown evidence.
- Completed third step:
  - `PR-SW03` added the CPU/reference proof harness
    (`tdmd/wavefront_reference.py`, `bench_wavefront_reference_equivalence.py`,
    `wavefront_reference_smoke`) and proved wave-batch equivalence against the current sequential
    slab CPU semantics for pair forces and current many-body target-local `EAM/eam-alloy`
    evaluation.
- Completed fourth step:
  - `PR-SW04` added CUDA runtime contract `pr_sw04_v1` in `tdmd/td_local.py`: admissible
    `1D` slab waves now fuse the pre-force half-step on CUDA, while zone consumption remains
    sequential-per-zone and the formal core invariant `W<=1` stays intact.
- Completed fifth step:
  - `PR-SW05` closed the operator-side go/no-go loop.
  - Representative evidence pack
    `results/pr_sw05_slab_wavefront_evidence_gpu_r2/slab_wavefront_evidence_gpu.summary.json`
    is fully green (`ok_all=true`) and confirms that the current slab-wavefront track is a real
    single-GPU scaling win on the tracked workloads.
- Current representative interpretation:
  - strict-valid crack compare at common `z=18` shows `4.47x` TD-vs-space speedup,
  - crack equal-`z` sweep stays above `1.0x` for every `z=1..12`, peaking at `3.63x` for `z=12`,
  - the `10K` `EAM/eam-alloy` control sweep uses common-valid `z=1..8` and peaks at `2.23x`
    for `z=8`,
  - control breakdown at `z=8` shows `2.22x` TD-vs-space speedup with realized
    `wave_batch_launches_saved_per_step=6.0` and neighbor reuse about `33%`.
- Current outcome:
  - keep the slab-wavefront runtime path,
  - keep the advisor recommendation-only,
  - treat `pr_sw05_v1` diagnostics and `slab_wavefront_evidence_gpu` as the current operator
    evidence basis for future scaling work.
- Future ML-potential GPU scaling should reuse the same many-body locality and slab-wavefront
  contracts, still starting from CPU reference semantics.
- See `docs/CUDA_EXECUTION_PLAN.md` for the consolidated runbook and `docs/GPU_BACKEND_API.md`
  for the strict backend contract.
