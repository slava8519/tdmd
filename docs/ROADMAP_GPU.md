# GPU Roadmap (CUDA-Only)

Principle:
- GPU is compute-backend refinement only.
- TD automaton/control semantics stay on CPU reference path.

## Completed Baseline
- CUDA-first runtime path exists.
- Strict GPU presets exist (`gpu_smoke`, `gpu_interop_smoke`, `gpu_metal_smoke`, `gpu_smoke_hw`).

## Active Cycle
- CUDA execution cycle: `PR-C01 -> PR-C08`.
- Full plan: `docs/CUDA_EXECUTION_PLAN.md`.
- Compact queue: `docs/PR_PLAN_GPU.md`.

## Stack Policy
- Primary stack for near-term PRs: `numba-cuda`.
- Plan B for perf/prod hardening after stabilization:
  - `CuPy RawKernel` or
  - C++/CUDA extension.

## Acceptance Baseline (always)
- `.venv/bin/python -m pytest -q`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict`
- `.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --strict`
- GPU-touching work additionally requires strict GPU presets per `docs/VERIFYLAB.md`.
