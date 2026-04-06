# Universal Visualization Contract

Status: v4.9 visualization track is implemented (`PR-VZ01..PR-VZ08`).
Current maintenance refinement adds optional runtime telemetry JSONL for long-running/operator
workloads without changing solver semantics.

Visualization is a passive observability layer. It must not change TD scheduling, force semantics, or automaton states (`F/D/P/W/S`).

## Contract
- Baseline trajectory format: LAMMPS dump (`.lammpstrj`), optional gzip (`.lammpstrj.gz`).
- Baseline metrics format: CSV (`metrics.csv`).
- Optional telemetry format: JSONL (`telemetry.jsonl`) plus summary sidecar
  (`telemetry.jsonl.summary.json`).
- Both outputs can emit sidecar manifests (`*.manifest.json`) with explicit schema/version metadata.
- Column order is deterministic and capability-driven.

Trajectory mandatory columns:
- `id type x y z vx vy vz`

Optional trajectory channels:
- `unwrapped`: `xu yu zu`
- `image`: `ix iy iz`
- `force`: `fx fy fz` (snapshot force, passive export only)

Metrics columns:
- `step T E_kin E_pot P vmax buffer`

Telemetry fields (JSONL snapshots):
- `step`, `steps_total`, `step_fraction`, `wall_sec`, `delta_wall_sec`
- `steps_per_sec_avg`, `steps_per_sec_inst`, `eta_sec`
- `atom_count`, `box`, `coord_min`, `coord_max`
- `speed_mean`, `vmax`, `temperature_est`
- `rss_mb`, `peak_rss_mb`, `cpu_user_sec`, `cpu_system_sec`
- `device`, `mode`
- CUDA-only when available: `gpu_device_*`, `gpu_pool_*`

## Runtime Interface
CLI (`tdmd.main run`) supports the same output controls for `serial`, `td_local`, and `td_full_mpi`:
- `--traj`
- `--traj-every`
- `--traj-channels unwrapped,image,force,all`
- `--traj-compression none|gz`
- `--metrics`
- `--metrics-every`
- `--telemetry`
- `--telemetry-every`
- `--telemetry-heartbeat-sec`
- `--telemetry-stdout`
- `--no-output-manifest`

Example:
```bash
python -m tdmd.main run examples/td_1d_morse.yaml \
  --task examples/interop/task.yaml \
  --mode td_local --device cpu \
  --traj results/run/traj.lammpstrj.gz --traj-every 10 \
  --traj-channels unwrapped,image \
  --traj-compression gz \
  --metrics results/run/metrics.csv --metrics-every 10 \
  --telemetry results/run/telemetry.jsonl --telemetry-every 10 \
  --telemetry-heartbeat-sec 5 \
  --telemetry-stdout
```

Telemetry is a passive observability layer:
- it does not feed back into force evaluation, scheduling, or automaton decisions;
- it may be enabled independently of trajectory/metrics;
- it is intended for operator progress/resource visibility on long runs and benchmarks;
- optional heartbeats keep resource/progress visibility alive even when a single TD timestep is
  itself long-running.
- operator benchmark scripts may emit one telemetry stream per compared case; these sidecars use
  the same schema and are intended to answer "which step completed?" and "what resources were
  consumed?" after timeout or early termination, not just after successful completion.

## Post-Processing API
`tdmd/viz.py` provides a mode-agnostic analysis API:
- `iter_lammpstrj(...)`
- plugin interface (`BaseVizPlugin`)
- built-in plugins:
  - `mobility`
  - `species_mixing`
  - `region_occupancy` (includes crack-mouth fill style metrics)

Script entrypoint:
```bash
python scripts/viz_analyze.py --traj results/run/traj.lammpstrj.gz \
  --plugin mobility \
  --plugin species_mixing \
  --plugin region:xlo=0,xhi=10,ylo=0,yhi=10,zlo=0,zhi=10,label=mouth \
  --out-csv results/run/viz/analysis.csv \
  --out-json results/run/viz/analysis.json
```

## External Adapters
Implemented adapter scripts:
- `scripts/viz_ovito_adapter.py`
- `scripts/viz_vmd_adapter.py`
- `scripts/viz_paraview_adapter.py`
- orchestrator: `scripts/viz_bundle.py`

Bundle example:
```bash
python scripts/viz_bundle.py \
  --traj results/run/traj.lammpstrj.gz \
  --outdir results/run/viz
```

Adapters always emit JSON status artifacts. Missing external binaries are reported explicitly (`skipped_missing_*`) and do not silently pass as solver failures.

## Large-Scale I/O Policy
- Frame thinning: use `--traj-every` and `--metrics-every`.
- Streaming writes: trajectory and metrics are frame-streamed (bounded memory).
- Compression: use `--traj-compression gz` for large runs when storage pressure dominates.
- Keep manifests enabled in strict and reproducibility-focused runs.
- Prefer `--telemetry-every` values that are coarse enough to avoid I/O noise on very large runs
  while still giving actionable progress visibility.

## Verification Gate
Strict visualization contract preset:
```bash
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset viz_smoke --strict
```

`viz_smoke` runs `scripts/bench_viz_contract.py`, validates required artifacts/manifests, and checks bundle output deterministically.

## PR Status
- `PR-VZ01` output schema governance: done.
- `PR-VZ02` runtime output abstraction across run modes: done.
- `PR-VZ03` extended channels and CLI controls: done.
- `PR-VZ04` universal post-processing plugin API: done.
- `PR-VZ05` OVITO/VMD/ParaView adapters: done.
- `PR-VZ06` large-scale I/O policy (thinning/streaming/compression): done.
- `PR-VZ07` strict VerifyLab visualization gate (`viz_smoke`): done.
- `PR-VZ08` docs/prompt consolidation: done.

## Safety Checklist
- No changes to TD transition rules or readiness predicates.
- No runtime feedback loop from visualization into integration.
- No added global barriers for visualization features.
- New output fields are documented, manifest-versioned, and tested.
