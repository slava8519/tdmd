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
- Current interpretation of representative large-run evidence:
  - on one GPU, TD still beats space decomposition at equal zone count,
  - but absolute TD runtime currently worsens as `z` grows because per-zone orchestration,
    launch, and halo-duplication overhead grows faster than the implementation hides it,
  - therefore the next optimization target is not "more zones" by itself, but "cheaper execution
    of several formally independent slabs in one GPU wave".
- Planned follow-up order:
  1. formalize the slab-wavefront contract and observability vocabulary,
  2. keep the `100K` Al microcrack benchmark plus an `EAM/eam-alloy` control benchmark as
     representative evidence,
  3. prove CPU/reference wave-batch equivalence before CUDA runtime batching,
  4. implement fused CUDA wave execution for independent slabs,
  5. feed that corrected cost model into operator guidance and recommendation-only zoning advice.
- Go/no-go rule:
  if representative wavefront execution does not beat the best current sequential TD wall-clock
  on at least one large workload, keep the simpler `z ~ G` guidance and do not force the more
  complex runtime path.
- Future ML-potential GPU scaling should reuse the same many-body locality and slab-wavefront
  contracts, still starting from CPU reference semantics.
- See `docs/CUDA_EXECUTION_PLAN.md` for the consolidated runbook and `docs/GPU_BACKEND_API.md`
  for the strict backend contract.
