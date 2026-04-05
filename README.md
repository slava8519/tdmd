# tdmd

TDMD-TD is a research-grade **Time Decomposition (TD)** molecular dynamics implementation based on an academic dissertation.

Project goals:
- preserve formal TD semantics (states/transitions `F/D/P/W/S`) and invariants,
- keep **CPU as the reference** behavior,
- treat GPU execution as **refinement only** (parity within strict tolerances),
- target **metals and alloys** workflows (`eam/alloy`, multi-type, multi-mass).

Active cycle:
- CUDA execution cycle: `docs/CUDA_EXECUTION_PLAN.md` (`PR-C01..PR-C08`).

Versioning:
- GitHub releases/tags + `RELEASE_NOTES.md` (README is intentionally not a changelog).

## Install

Requirements:
- Python 3.11+
- `pip`
- MPI runtime for TD-MPI runs (e.g. OpenMPI) and for `mpi4py` installation on Linux.

Setup:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Quick workstation bootstrap (Ubuntu/Debian):
```bash
bash scripts/bootstrap_codex_env.sh
```

Optional GPU (current implementation is CUDA-first via CuPy):
- install a CuPy build matching your CUDA version, then use `--device cuda` (falls back to CPU if unavailable).

All commands below assume `source .venv/bin/activate`.

## Quickstart

CPU only (serial), 5 commands:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m tdmd.main run examples/td_1d_morse.yaml --mode serial --device cpu
```

MPI (TD-MPI), 2 and 4 ranks:
```bash
mpirun -np 2 python -m tdmd.main run examples/td_1d_morse_static_rr.yaml --mode td_full_mpi --device cpu
mpirun -np 4 python -m tdmd.main run examples/td_1d_morse_static_rr_smoke4.yaml --mode td_full_mpi --device cpu
```

GPU (CUDA):
```bash
# Prereq: install a CuPy build matching your CUDA version (example for CUDA 12.x):
# python -m pip install cupy-cuda12x
python -m tdmd.main run examples/td_1d_morse.yaml --mode td_local --device cuda
```

Notes:
- If `--device cuda` is requested but unavailable, TDMD falls back to CPU with a warning.
- Hardware-strict GPU verification must treat CPU fallback as failure (see `docs/VERIFYLAB_GPU.md`).

## Quick Checks (Strict)

Recommended local gates (CI-grade):
```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict
.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --strict
.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset eam_decomp_perf_smoke --strict
```

Make targets:
```bash
make test
make verify-smoke
```

## Run

TDMD supports three execution modes:
- `serial`: CPU reference integrator (no TD scheduling).
- `td_local`: TD scheduling on a single process (sync/async semantics).
- `td_full_mpi`: TD-MPI ring-transfer/ownership (requires `mpirun` + `mpi4py`).

### Run From Config (YAML)

Serial (CPU):
```bash
python -m tdmd.main run examples/td_1d_morse.yaml --mode serial --device cpu
```

TD-local (CPU):
```bash
python -m tdmd.main run examples/td_1d_morse.yaml --mode td_local --device cpu
```

TD-MPI (CPU, 4 ranks):
```bash
mpirun -np 4 python -m tdmd.main run examples/td_1d_morse_static_rr_smoke4.yaml --mode td_full_mpi --device cpu
```

Notes:
- `td_full_mpi` supports `NVT` generally; `NPT` is guarded to `MPI size=1` (see `docs/MODE_CONTRACTS.md`).

### Run From Task (Interop YAML)

Task-driven initialization:
```bash
python -m tdmd.main run --task examples/interop/task.yaml --mode td_local --device cpu
```

EAM/alloy examples (CPU reference):
```bash
python -m tdmd.main run --task examples/interop/task_eam_al.yaml --mode serial --device cpu
python -m tdmd.main run --task examples/interop/task_eam_alloy.yaml --mode serial --device cpu
```

### Outputs (Trajectory + Metrics + Manifests)

Trajectory and metrics with schema sidecars:
```bash
python -m tdmd.main run --task examples/interop/task.yaml --mode td_local --device cpu \
  --traj results/viz_demo/traj.lammpstrj.gz --traj-every 10 \
  --traj-channels unwrapped,image --traj-compression gz \
  --metrics results/viz_demo/metrics.csv --metrics-every 10
```

## Verification

The verification entrypoint is VerifyLab matrix:
- docs: `docs/VERIFYLAB.md`
- runner: `scripts/run_verifylab_matrix.py`

Common strict presets:
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --strict
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset eam_decomp_perf_smoke --strict
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset metal_smoke --strict
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_metal_smoke --strict
```

`eam_decomp_perf_smoke` is the standard PR benchmark for `EAM/alloy`. It prints a four-column
table (`space_cpu`, `space_gpu`, `time_cpu`, `time_gpu`) and derived speedup rows after the run.

For a heavier manual GPU-only benchmark on `EAM/eam-alloy`, comparing TD against space
decomposition on a `10K`-atom model, run:
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset eam_decomp_perf_gpu_heavy --strict
```

This preset runs only `space_gpu` and `time_gpu`, requires effective CUDA execution, and is meant
for operator-side performance evaluation rather than PR CI.

To sweep several GPU zone layouts on the same `10K`-atom `EAM/eam-alloy` model and compare
`time_gpu` against `space_gpu`, run:
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset eam_decomp_zone_sweep_gpu --strict
```

This sweep is also operator-side only. It is intended to identify favorable zone counts/layouts
before any future automation of TD zoning policy.

To see where GPU time is currently spent for the best observed `10K`-atom `EAM/eam-alloy` TD
layout, run:
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset eam_td_breakdown_gpu --strict
```

This manual benchmark compares `space_gpu` and `time_gpu` at the same `2-zone` layout and emits a
breakdown for `forces_full`, device sync, cell-list build, candidate enumeration, and zone
bookkeeping. Use it before making claims about the TD optimization ceiling or changing zoning
policy.

For a broader CUDA-cycle profiling report, run:
```bash
python scripts/profile_gpu_backend.py --config examples/td_1d_morse.yaml --out-csv results/gpu_profile.csv --out-md results/gpu_profile.md --out-json results/gpu_profile.summary.json --require-effective-cuda --eam-n-atoms 512 --eam-steps 2
```

This consolidated profiler keeps the old CPU-vs-GPU preset timing table and additionally records:
- `gpu_perf_smoke` transfer/kernel metrics,
- current `EAM/alloy` decomposition speedups,
- optional `Phase E` comparison when you pass `--phase-e-worktree <path>`.

Historical baseline note:
- `Phase E` reference commit for CUDA-cycle comparisons is `efb864e` (last pre-PR-C02 high-level CuPy baseline).
- Example temporary worktree:
  `git worktree add /tmp/tdmd_phase_e efb864e`

GPU verification docs:
- `docs/GPU_BACKEND.md`
- `docs/VERIFYLAB_GPU.md`

## Visualization

Visualization is a passive observability layer:
- governance: `docs/VISUALIZATION.md`
- strict gate: `viz_smoke`

Bundle artifacts for OVITO/VMD/ParaView:
```bash
python scripts/viz_bundle.py --traj results/viz_demo/traj.lammpstrj.gz --outdir results/viz_demo/viz
```

## Project Layout

Top-level structure:
- `tdmd/`: core runtime (TD automaton, integrators, potentials, I/O).
- `scripts/`: VerifyLab, benchmarks (cluster/transport), visualization adapters.
- `tests/`: unit/regression tests (strict gates live here + in VerifyLab).
- `examples/`: config YAMLs and interop task fixtures.
- `golden/`: strict baselines and threshold policy artifacts.
- `docs/`: specs, invariants, mode contracts, governance.
- `results/`: run artifacts (ignored by git).

## Documentation Map

Start here:
- `AGENTS.md` (roles, strict acceptance, ownership)
- `CODEX_ENV_BOOTSTRAP_PROMPT.md` (fresh-machine Codex bootstrap prompt)
- `docs/TODO.md` (PR-sized tasks, critical order)
- `docs/CUDA_EXECUTION_PLAN.md` (active CUDA execution cycle)
- `docs/MODE_CONTRACTS.md` (mode guarantees and required gates)

Formal semantics:
- `docs/SPEC_TD_AUTOMATON.md` (TD automaton mapping to code)
- `docs/INVARIANTS.md` (invariants and concrete checks)
- `docs/THEORY_VS_CODE.md` (where theory lands in modules)

Verification governance:
- `docs/VERIFYLAB.md`
- `docs/GPU_BACKEND.md`
- `docs/VERIFYLAB_GPU.md`

Visualization contract:
- `docs/VISUALIZATION.md`

## Contributing

See `CONTRIBUTING.md`.


## Liveness
- `docs/LIVENESS.md` defines progress / deadlock-avoidance lemmas for general deps-graphs.


## Documentation
- Project status: `docs/PROJECT_STATUS.md`
- Versioning: `docs/VERSIONING.md`
- GPU roadmap: `docs/ROADMAP_GPU.md`
- GPU PR plan: `docs/PR_PLAN_GPU.md`
- Codex runbook: `docs/CODEX_RUNBOOK.md`
- Codex env bootstrap script: `scripts/bootstrap_codex_env.sh`

- Contracts (do-not-break rules): `docs/CONTRACTS.md`
