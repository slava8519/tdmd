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
- Current large-run `EAM/eam-alloy` evidence shows single-GPU TD remains dominated by repeated
  many-body `forces_full`, so the next algorithmic step is target-local many-body TD evaluation
  rather than auto-zoning first.
- `PR-MB01` is complete: the current `td_local` many-body force scope is explicit in code and in
  `eam_td_breakdown_gpu` artifacts (`pr_mb01_v1`), including the distinction between
  full-system evaluation and target-only consumption.
- `PR-MB02` is complete: CPU async `td_local` many-body now uses target-local
  `potential.forces_on_targets(...)` while preserving TD scheduling semantics.
- `PR-MB03` is complete: CUDA async `td_local` many-body now uses target/candidate-local GPU
  dispatch, and `eam_td_breakdown_gpu` reports current contract `pr_mb03_v1` against baseline
  `pr_mb01_v1` with `forces_full_share=0` on the representative `10K` benchmark.
- `PR-ZA01` is complete: `td_autozoning_advisor_gpu` now detects available CPU/GPU/MPI
  resources, enumerates strict-valid `TD/space` layouts, and emits recommendation-only
  markdown/json/csv artifacts without changing runtime zoning policy.
- `PR-ML01` is complete: `potential.kind=ml/reference` now formalizes a versioned CPU-reference
  many-body contract (`cutoff`, `descriptor`, `neighbor`, `inference`) with explicit
  no-hidden-barrier semantics and target-local `forces_on_targets(...)` support for TD runtimes.
- `PR-ML02` is complete: the `quadratic_density` `ml/reference` family now has both a strict
  task-based VerifyLab lane (`ml_reference_smoke`) and a versioned fixture-driven parity suite
  (`examples/interop/ml_reference_suite_v1.json` via `scripts/ml_reference_parity_pack.py`).
- Planned follow-up order:
  1. keep any future ML work on CPU-reference + explicit-contract discipline until a real
     scientific model family is chosen for expansion,
  2. keep any future auto-zoning automation recommendation-first until an explicit runtime
     contract is defined and verified.
- Future ML-potential work should reuse the same many-body locality contract and still start from
  CPU reference semantics before any GPU or zoning policy extension.
- See `docs/CUDA_EXECUTION_PLAN.md` for the consolidated runbook and `docs/GPU_BACKEND_API.md`
  for the strict backend contract.
