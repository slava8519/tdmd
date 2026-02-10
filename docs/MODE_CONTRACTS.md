# Mode Contracts

This document defines post-hardening behavioral contracts by execution mode.
It is governance-level policy and does not change TD automaton semantics.
Active next-cycle extensions (cluster strict lanes, materials property-level strict lanes) are tracked in `docs/RISK_BURNDOWN_PLAN_V2.md` and become mandatory once implemented.
Next active portability cycle is tracked in `docs/PORTABILITY_KOKKOS_PLAN.md` (`PR-K01..PR-K10`).

## Global Contract
- TD states/transitions `F/D/P/W/S` are preserved.
- No implicit global barriers are introduced.
- CPU path is formal reference semantics.
- GPU path is refinement only (numerical parity within documented tolerances).
- Current implementation is CUDA-first; planned portability extension (NVIDIA + AMD via Kokkos) must preserve the same semantic contract.
- Visualization/output pipelines are passive observability layers and must not feed back into runtime scheduling/forces.
- Runtime ensemble support:
  - `serial`: `NVE/NVT/NPT`
  - `td_local`: `NVE/NVT/NPT`
  - `td_full_mpi`: `NVE/NVT` generally, `NPT` with `MPI size=1`

## Mode Matrix

| Mode | Semantic guarantee | Numeric guarantee | Required strict gates |
|---|---|---|---|
| `serial` | Reference integrator path, no TD scheduling | Reference baseline for comparisons | `smoke_ci`, materials checks when touched |
| `td_local` (`sync_mode=True`) | TD decomposition path with synchronous snapshot update | Serial parity target for configured thresholds | `smoke_ci` and task/material strict presets as applicable |
| `td_local` (`sync_mode=False`) | Asynchronous TD scheduling semantics preserved | Not bitwise/stepwise serial-equivalent; bounded by envelope policy | `longrun_envelope_ci` for long-horizon drift control |
| `td_full_mpi` (`dynamic`) | Ring-transfer core semantics preserved | Serial closeness is tolerance-based, not identity | strict smoke + MPI overlap strict presets when transport changes |
| `td_full_mpi` (`static_rr`) | Fixed ownership extension, no ownership transfer in steady-state | Serial closeness is tolerance-based, not identity | strict smoke + MPI overlap strict presets when transport changes |
| GPU `serial`/`td_local` (current CUDA-first) | Same control semantics as CPU mode | CPU-equivalent within tolerance, fallback explicit | `gpu_smoke`; hardware validation via `*_hw` |
| GPU `td_full_mpi` / multi-GPU (current CUDA-first) | Same TD-MPI semantics with rank->device mapping | CPU-equivalent within tolerance; transport strategy is refinement | `gpu_*` strict presets + overlap strict presets for transport changes |
| GPU portability extension (planned Kokkos CUDA/HIP) | Must preserve same TD control semantics as above | Same tolerance policy; hardware-strict no-fallback remains mandatory | Existing `gpu_*` strict gates + planned vendor lanes per `docs/PORTABILITY_KOKKOS_PLAN.md` |

## Non-Guarantees
- Bitwise equality across modes/backends is not guaranteed.
- Async modes (`td_local` async, `td_full_mpi`) are not required to match serial at every intermediate step.
- Performance metrics (speedup/latency) are diagnostic unless explicitly gated by a benchmark policy.

## Gate Mapping
- Baseline strict CI:
  - `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict`
  - `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --strict`
  - `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset nvt_smoke --strict`
  - `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset npt_smoke --strict`
- Materials strict:
  - `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset metal_smoke --strict`
  - `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_metal_smoke --strict`
  - `python scripts/materials_parity_pack.py --fixture examples/interop/materials_parity_suite_v1.json --config examples/td_1d_morse.yaml --strict`
- Long-run async envelope:
  - `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset longrun_envelope_ci --strict`
- Cluster validation (v2):
  - `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset cluster_scale_smoke --strict`
  - `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset cluster_stability_smoke --strict`
  - `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset mpi_transport_matrix_smoke --strict`
- GPU strict / hardware strict:
  - `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_smoke --strict`
  - `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_interop_smoke --strict`
  - `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_metal_smoke --strict`
  - `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_smoke_hw --strict`
- GPU portability cycle (planned additions, in addition to existing gates):
  - backend-agnostic hardware-strict policy key: `require_effective_gpu` (with compatibility alias support during migration),
  - vendor lanes: `gpu_cuda_smoke_hw`, `gpu_hip_smoke_hw`,
  - cross-vendor parity lane: `gpu_portability_smoke`.
- MPI overlap/cuda-aware strict:
  - `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset mpi_overlap_smoke --strict`
  - `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset mpi_overlap_cudaaware_smoke --strict`
- Materials property strict (v2):
  - `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset metal_property_smoke --strict`
  - `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_metal_property_smoke --strict`
- Visualization contract strict:
  - `python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset viz_smoke --strict`
  - required for visualization/output-contract changes.

## Cluster Profile Contract
- Cluster verification scripts are profile-driven (`examples/cluster/*.yaml`).
- `cluster_profile_smoke.yaml` is CI-safe and permits simulated execution when MPI launcher/hardware is unavailable.
- Hardware cluster validation must use profiles with:
  - `allow_simulated_cluster=false`,
  - `prefer_simulated=false`.
- Scale-envelope governance is versioned via `golden/cluster_scale_envelope_v1.json`.

## Failure Triage Contract
- Any strict VerifyLab failure should produce a reproducible incident bundle:
  - automatic in `scripts/run_verifylab_matrix.py` strict-fail path,
  - manual export: `python scripts/export_incident_bundle.py --run-dir results/<run_id> --reason manual_export`.


## Liveness gate (deps-graph)
For any non-ring deps-graph mode, the runtime must satisfy `LIVENESS` assumptions A1–A4.
In strict mode: either prove/enforce `WFG(t)` acyclic (A4a) or implement deterministic cycle-breaking (A4b).
