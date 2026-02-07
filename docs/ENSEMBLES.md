# Ensemble Contract (NVE/NVT/NPT)

This document defines schema + runtime behavior for ensemble control.
It does not change TD automaton states/transitions (`F/D/P/W/S`).

## Schema
Task (`tdmd/io/task.py`) and config (`tdmd/config.py`) support:

```yaml
ensemble:
  kind: nve | nvt | npt
  thermostat:   # required for nvt/npt
    kind: <string>
    params: { ... }
  barostat:     # required for npt
    kind: <string>
    params: { ... }
```

Strict validation rules:
- `kind=nve`: thermostat/barostat must be absent.
- `kind=nvt`: thermostat required, barostat forbidden.
- `kind=npt`: thermostat and barostat both required.
- Unknown keys inside `ensemble`, `ensemble.thermostat`, `ensemble.barostat` are rejected.

## Runtime Support Matrix
- `serial`: `NVE`, `NVT`, `NPT`.
- `td_local`: `NVE`, `NVT`, `NPT`.
- `td_full_mpi`:
  - `NVE`, `NVT` for general MPI size.
  - `NPT` currently requires `MPI size=1` (guardrail to avoid introducing cross-rank global barostat synchronization).

Legacy behavior:
- top-level task field `thermostat` is parsed as backward-compatibility alias and kept non-executable.
- mixing legacy top-level `thermostat` with `ensemble` is rejected as ambiguous.

## Implemented Controllers (CPU reference)
- Thermostat: `berendsen`.
- Barostat: `berendsen` isotropic.

Supported controller params:
- Thermostat (`ensemble.thermostat.params`):
  - `t_target` (required),
  - `tau` (required),
  - `every` (optional, default `1`),
  - `max_scale_step` (optional, default `0.2`).
- Barostat (`ensemble.barostat.params`):
  - `p_target` (required),
  - `tau` (required),
  - `compressibility` (optional, default `1e-4`),
  - `every` (optional, default `1`),
  - `max_volume_scale_step` (optional, default `0.1`),
  - `scale_velocities` (optional, default `true`).

## LAMMPS Interop
`tdmd/io/lammps.py::export_lammps_in` now emits integration `fix` based on ensemble:
- `NVE`: `fix ... nve`
- `NVT`: `fix ... nvt temp ...`
- `NPT`: `fix ... npt temp ... iso ...`

This keeps exported scripts self-contained for quick parity runs.

## Semantics Boundary
- Ensemble controls are applied as post-step velocity/box refinements.
- TD automaton event graph is unchanged.
- Existing invariants (`hG3/hV3/tG3`, `violW`, `lagV`) remain unchanged in definition.
