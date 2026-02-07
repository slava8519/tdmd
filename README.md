# tdmd

TDMD-TD is a research-grade **Time Decomposition (TD)** molecular dynamics implementation based on an academic dissertation.

Project goals:
- preserve formal TD semantics (states/transitions `F/D/P/W/S`) and invariants,
- keep **CPU as the reference** behavior,
- treat GPU execution as **refinement only** (parity within strict tolerances),
- target **metals and alloys** workflows (`eam/alloy`, multi-type, multi-mass).

Active cycle:
- GPU portability (NVIDIA + AMD) via Kokkos: `docs/PORTABILITY_KOKKOS_PLAN.md` (`PR-K01..PR-K10`).

Versioning:
- GitHub releases/tags + `RELEASE_NOTES.md` (README is intentionally not a changelog).

## Install

Requirements:
- Python 3.11+
- `pip`
- MPI runtime for TD-MPI runs (e.g. OpenMPI) and for `mpi4py` installation on Linux.

Setup:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional GPU (current implementation is CUDA-first via CuPy):
- install a CuPy build matching your CUDA version, then use `--device cuda` (falls back to CPU if unavailable).

## Quick Checks (Strict)

Recommended local gates (CI-grade):
```bash
python -m pytest -q
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --strict
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
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset metal_smoke --strict
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_metal_smoke --strict
```

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
- `docs/TODO.md` (PR-sized tasks, critical order)
- `docs/PORTABILITY_KOKKOS_PLAN.md` (active GPU portability cycle)
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
