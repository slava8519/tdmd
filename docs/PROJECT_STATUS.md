# Project status (as of v4.5.4)

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
- **Test suite**: 169 tests (28 added for `constants` and `zones` modules).

## GPU Backend Status
- **Phase E (complete)**: CuPy high-level API path — functionally correct, all strict gates pass.
  Known performance limitations: O(N²) broadcast matrices, Python for-loops in cell-list
  and EAM paths, no persistent GPU state.
- **Phase H (active)**: CUDA execution cycle migrating to **CuPy RawKernel** — real CUDA C
  kernels for neighbor lists, pair forces, EAM, with O(N) memory and single-launch execution.
  See `docs/CUDA_EXECUTION_PLAN.md` for full PR queue (PR-C02..C08).
- **Stack decision**: CuPy RawKernel chosen over Numba-CUDA. CuPy is already a dependency;
  RawKernel provides full CUDA C control without adding a new runtime. Plan B: C++/CUDA
  extension if RawKernel performance ceiling is reached.

## Next
- Execute PR-C02 (GPU neighbor list kernel via CuPy RawKernel).
- See `docs/CUDA_EXECUTION_PLAN.md` for the active cycle.
- See `docs/GPU_BACKEND_API.md` for the strict backend contract.
