# CLAUDE.md — AI Agent Instructions for TDMD-TD

## Project

TDMD-TD is a research-grade **Time Decomposition (TD)** molecular dynamics
implementation based on an academic dissertation. Current version: see `VERSION`.

Stack: Python 3.11+, NumPy, PyYAML, mpi4py, optional CuPy (CUDA).
Target domain: metals and alloys (EAM/eam-alloy, multi-type, multi-mass).

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
make test           # pytest -q (239 tests)
make verify-smoke   # pytest + smoke_ci + interop_smoke + eam_decomp_perf strict
```

## Non-Negotiable Rules

1. **TD semantics are sacred.** Do NOT bypass/merge TD states (F/D/P/W/S).
   Keep the formal core invariant W<=1 intact.
2. **CPU is the reference.** GPU paths are compute-backend refinements only.
   GPU must stay numerically aligned with CPU under defined tolerances.
3. **No implicit barriers.** Do not assume synchronous execution or global
   barriers unless explicitly modeled.
4. **No false green.** Non-strict diagnostic runs do not count as acceptance.
   For hardware-strict GPU tasks, CPU fallback is failure.
5. **Any new mechanism must** preserve existing invariants, or introduce a new
   invariant + verification (unit test or VerifyLab metric/counter).
6. **Prefer minimal diffs** and PR-sized changes. One change per PR.
7. **Visualization is passive.** No feedback into integrator/automaton decisions.

## Architecture

| Module | Purpose |
|---|---|
| `tdmd/td_automaton.py` | TD state machine (F/D/P/W/S transitions) |
| `tdmd/serial.py` | CPU reference integrator (no TD) |
| `tdmd/td_local.py` | Single-process TD scheduling |
| `tdmd/td_full_mpi.py` | TD-MPI ring-transfer/ownership |
| `tdmd/forces_gpu.py` | CUDA force kernels (CuPy RawKernel) |
| `tdmd/potentials.py` | Force fields (Morse, table, EAM/eam-alloy) |
| `tdmd/zones.py` | Zone layout and assignment |
| `tdmd/verify_v2.py` | Verification engine |
| `tdmd/verify_lab.py` | VerifyLab matrix runner |
| `tdmd/io/` | Task schema, LAMMPS I/O, metrics, telemetry, trajectory |
| `tdmd/backend.py` | CPU/CUDA backend dispatch |
| `tdmd/config.py` | YAML config parsing |
| `scripts/` | VerifyLab presets, benchmarks, profiling, visualization |
| `tests/` | Unit/regression tests |
| `golden/` | Strict baselines and threshold policies |

## Quality Gates

**Always required (every PR):**
```bash
make test                    # or: .venv/bin/python -m pytest -q
make verify-smoke            # smoke_ci + interop_smoke + eam_decomp_perf strict
```

**Conditional gates (run when relevant):**

| Scope | Gate |
|---|---|
| GPU | `gpu_smoke`, `gpu_interop_smoke`, `gpu_metal_smoke` |
| GPU hardware-strict | `gpu_smoke_hw` (rejects CPU fallback) |
| Ensemble | `nvt_smoke`, `npt_smoke` |
| Materials/potentials | `metal_smoke`, `interop_metal_smoke`, `materials_parity_pack --strict` |
| Materials properties | `metal_property_smoke`, `interop_metal_property_smoke` |
| MPI/cluster | `cluster_scale_smoke`, `cluster_stability_smoke`, `mpi_transport_matrix_smoke` |
| Visualization | `viz_smoke` |
| Long-horizon | `longrun_envelope_ci` |
| ML reference | `ml_reference_smoke`, `ml_reference_parity_pack.py --strict` |

## Code Style

- **black** (line-length 100, skip-string-normalization)
- **ruff** (rules: I, F, UP, B)
- **mypy** typed island: `atoms.py`, `output.py`, `observer.py`, `force_dispatch.py`, `cli_parser.py`
- Naming: snake_case functions/variables, PascalCase classes
- Run `make fmt` and `make lint` before committing

## Key Documentation

| Document | Purpose |
|---|---|
| `AGENTS.md` | Agent roles and ownership scopes |
| `docs/TODO.md` | PR-sized task queue (critical order) |
| `docs/PROJECT_STATUS.md` | Current state and active track |
| `docs/CUDA_EXECUTION_PLAN.md` | GPU/post-CUDA execution plan |
| `docs/MODE_CONTRACTS.md` | Mode guarantees and required gates |
| `docs/SPEC_TD_AUTOMATON.md` | TD automaton formal spec |
| `docs/INVARIANTS.md` | Invariants and concrete checks |
| `docs/THEORY_VS_CODE.md` | Theory-to-module mapping |
| `docs/VERIFYLAB.md` | Verification framework governance |
| `docs/GPU_BACKEND.md` | GPU backend architecture |
| `docs/CONTRACTS.md` | Do-not-break contracts |

## Current Development Focus

Next active maintenance theme: single-GPU **1D slab-wavefront** TD scaling
(PR-SW01..SW05) for large metals/alloys runs. Goal: allow multiple formally
independent slab zones in one GPU wave without implicit barriers or semantic
drift. See `docs/CUDA_EXECUTION_PLAN.md` and `docs/TODO.md`.

## GPU Stack Policy

- Primary: CuPy RawKernel
- Decision: stay on CuPy RawKernel
- Plan B (C++/CUDA extension): only with fresh `scripts/profile_gpu_backend.py` evidence
