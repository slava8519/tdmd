# Current Worktree Slice Plan

Purpose: split the current dirty worktree into a small set of commit-ready PR slices without
mixing core CUDA runtime changes, observability/benchmark work, and post-cycle governance.

This is a tactical slicing plan for the current local worktree, not a replacement for
`docs/PR_PLAN_GPU.md` or `docs/TODO.md`.

## Order
1. Slice A: CUDA runtime hardening
2. Slice B: EAM decomposition benchmarks + VerifyLab wiring
3. Slice C: GPU perf smoke calibration + consolidated profiler
4. Slice D: governance / roadmap / prompt cleanup

## Shared-file Warning
- A few files contain mixed hunks from more than one slice:
  - `scripts/run_verifylab_matrix.py`
  - `README.md`
  - `RELEASE_NOTES.md`
  - `docs/GPU_BACKEND.md`
  - `docs/VERIFYLAB.md`
  - `docs/VERIFYLAB_GPU.md`
- When splitting commits, group these files by topic/hunk, not by whole-file ownership.
- The cleanest dependency chain is `A -> B -> C -> D`.

## Slice A - CUDA Runtime Hardening

Intent:
- Keep the real CUDA backend/runtime improvements together.
- This is the slice that should not be dropped lightly.

Primary outcomes:
- direct `table` GPU path uses neighbor-list pipeline instead of dense fallback,
- direct `EAM/eam-alloy` GPU path uses device-side candidate-union many-body work,
- persistent device state + dirty-range sync for runtime-managed CUDA paths,
- TD-MPI CUDA path uses the same dirty-range runtime policy,
- RawKernel handles and device potential tables are cached.

Files:
- `tdmd/forces_gpu.py`
- `tdmd/force_dispatch.py`
- `tdmd/serial.py`
- `tdmd/td_local.py`
- `tdmd/td_full_mpi.py`
- `tests/test_gpu_pair_kernels.py`
- `tests/test_td_full_mpi_gpu_refinement.py`

Relevant doc hunks:
- runtime notes in `docs/GPU_BACKEND.md`
- runtime notes in `docs/VERIFYLAB_GPU.md`
- runtime release-note entries in `RELEASE_NOTES.md`

Required gates:
- `.venv/bin/python -m pytest -q`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset metal_smoke --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_metal_smoke --strict`
- `.venv/bin/python scripts/materials_parity_pack.py --fixture examples/interop/materials_parity_suite_v1.json --config examples/td_1d_morse.yaml --strict`
- `.venv/bin/python scripts/materials_parity_pack.py --fixture examples/interop/materials_parity_suite_v2.json --config examples/td_1d_morse.yaml --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_smoke --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_interop_smoke --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_metal_smoke --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_smoke_hw --strict`
- If TD-MPI/runtime overlap hunks are included: overlap/cluster/transport strict lanes too.

## Slice B - EAM Decomposition Benchmarks + VerifyLab Wiring

Intent:
- Keep the new EAM TD-vs-space observability surface together.
- This slice is useful for operator evidence and PR perf reporting, but it is not required for
  runtime correctness.

Primary outcomes:
- standard PR benchmark `eam_decomp_perf_smoke`,
- heavy GPU-only `10K` benchmark,
- manual zone sweep benchmark,
- manual TD runtime breakdown benchmark,
- CI/Make targets and README/VerifyLab wiring for these reports.

Files:
- `scripts/bench_eam_decomp_perf.py`
- `scripts/bench_eam_zone_sweep_gpu.py`
- `scripts/bench_eam_td_breakdown_gpu.py`
- `tests/test_eam_decomp_perf_script.py`
- `tests/test_eam_zone_sweep_gpu_script.py`
- `tests/test_eam_td_breakdown_gpu_script.py`
- `tests/test_verifylab_eam_decomp_perf.py`
- benchmark/preset hunks in `scripts/run_verifylab_matrix.py`
- `Makefile`
- `.github/workflows/ci.yml`

Relevant doc hunks:
- benchmark sections in `README.md`
- EAM benchmark sections in `docs/VERIFYLAB.md`
- operator benchmark notes in `docs/VERIFYLAB_GPU.md`
- operator benchmark notes in `docs/GPU_BACKEND.md`
- benchmark release-note entries in `RELEASE_NOTES.md`

Required gates:
- `.venv/bin/python -m pytest -q`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_smoke --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_interop_smoke --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_metal_smoke --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_smoke_hw --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset eam_decomp_perf_smoke --strict`

## Slice C - GPU Perf Smoke Calibration + Consolidated Profiler

Intent:
- Keep micro-perf guardrail stabilization and stack-policy profiling together.
- This slice depends on Slice B because the profiler consumes the EAM benchmark script.

Primary outcomes:
- `gpu_perf_smoke` uses calibrated kernel timing floor,
- profiler writes markdown/json and optional Phase E comparison,
- profiler produces a single stack-policy decision artifact.

Files:
- `scripts/bench_gpu_perf_smoke.py`
- `scripts/profile_gpu_backend.py`
- profiler/calibration hunks in `scripts/run_verifylab_matrix.py`
- `tests/test_gpu_perf_smoke_script.py`
- `tests/test_gpu_profile_script.py`

Relevant doc hunks:
- `gpu_perf_smoke` calibration notes in `docs/VERIFYLAB.md`
- profiler notes in `docs/VERIFYLAB_GPU.md`
- operator playbook/profiler notes in `docs/GPU_BACKEND.md`
- profiler command/docs in `README.md`
- calibration/profiler release-note entries in `RELEASE_NOTES.md`

Required gates:
- `.venv/bin/python -m pytest -q`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_smoke --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_interop_smoke --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_metal_smoke --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_smoke_hw --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_perf_smoke --strict`
- `.venv/bin/python scripts/profile_gpu_backend.py --config examples/td_1d_morse.yaml --out-csv results/gpu_profile.csv --out-md results/gpu_profile.md --out-json results/gpu_profile.summary.json --require-effective-cuda`

## Slice D - Governance / Roadmap / Prompt Cleanup

Intent:
- Land the post-CUDA governance state only after the code and observability slices are in place.
- This slice also carries the new maintenance order:
  many-body TD locality first, auto-zoning second, ML-potential work later on the same contract.

Files:
- `AGENTS.md`
- `CODEX_GPU_MASTER_PROMPT.md`
- `docs/CUDA_EXECUTION_PLAN.md`
- `docs/GPU_BACKEND_API.md`
- `docs/PROJECT_STATUS.md`
- `docs/PR_PLAN_GPU.md`
- `docs/ROADMAP_GPU.md`
- `docs/TODO.md`

Optional shared doc hunks to land here if they are still pending:
- top-level governance wording in `README.md`
- top-level release summary in `RELEASE_NOTES.md`

Required gates:
- docs-only minimum: `.venv/bin/python -m pytest -q`
- recommended: `smoke_ci --strict` and `interop_smoke --strict` after the final assembled state

## Drop Policy
- Do not drop Slice A unless you intentionally want to roll back CUDA runtime progress.
- Slice B and Slice C are the only realistic drop candidates if you want to keep runtime changes but
  postpone benchmark/profiling surfacing.
- Slice D should match whatever code/observability slices you keep; do not leave governance claiming
  features that were dropped.
