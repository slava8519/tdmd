# GPU Backend

CUDA execution governance reference: `docs/CUDA_EXECUTION_PLAN.md` (PR-C01..PR-C08, CuPy RawKernel).

## Scope
GPU support is implemented as a refinement path over the CPU reference semantics.
The TD automaton (F/D/P/W/S transitions), dependency predicates, and invariants remain unchanged.
Mode-level guarantees and strict-gate ownership are documented in `docs/MODE_CONTRACTS.md`.

## Historical Milestone (PR-G01)
- Added backend selection abstraction in `tdmd/backend.py`.
- Supported device selectors:
  - `auto`
  - `cpu`
  - `cuda`
- Added config field `run.device` (`auto|cpu|cuda`) and CLI override `tdmd run --device ...`.
- `cuda` selection is capability-aware:
  - if CUDA/CuPy is available, backend resolves to `cuda`;
  - otherwise runtime warns and falls back to `cpu`.
- `TDMD_FORCE_CPU=1` forces CPU when `run.device=auto`.

## Completed CUDA Cycle (CuPy RawKernel)
- Migrated from CuPy high-level API to CuPy RawKernel (real CUDA C kernels).
- Key improvements: O(N) neighbor lists, single-launch force kernels, fused EAM,
  persistent GPU state, CUDA stream overlap.
- Stack: CuPy RawKernel primary; C++/CUDA extension as Plan B.
- Numba-CUDA is not in scope.
- Full plan: `docs/CUDA_EXECUTION_PLAN.md`.

## Semantics Guarantee
- CPU path remains the formal reference.
- Backend selection does not alter TD scheduling semantics.
- No global barrier is introduced by backend selection.

## Notes
- Historical `PR-G0x` sections below are retained as provenance for the archived `Phase E`
  high-level CuPy baseline.
- Profiling helper: `scripts/profile_gpu_backend.py` writes a consolidated CUDA-cycle report:
  - preset timing/parity artifacts (`results/gpu_profile.csv`),
  - operator markdown summary (`results/gpu_profile.md`),
  - machine-readable summary (`results/gpu_profile.summary.json`),
  - nested `gpu_perf_smoke` and `EAM/alloy` benchmark artifacts.

## Operator Playbook
1. Keep CPU as the formal reference and treat every GPU change as refinement-only.
2. Run hardware-strict CUDA acceptance whenever backend/runtime behavior is touched:
   - `gpu_smoke_hw`
   - baseline `gpu_smoke`, `gpu_interop_smoke`, `gpu_metal_smoke`
3. Use `make gpu-profile` or the explicit `scripts/profile_gpu_backend.py` command before any
   stack-policy discussion.
4. Treat `gpu_perf_smoke` as a micro-perf warning signal, not as the sole Plan B trigger.
   The smoke benchmark calibrates its synthetic kernel to a minimum timing floor before applying
   `transfer_over_kernel`, so the signal is less sensitive to sub-millisecond timer jitter.
5. Use representative `EAM/alloy` profiling (`512 atoms`, `2 steps`) plus optional `Phase E`
   comparison against `efb864e` for Plan B decisions.
6. For zone-layout decisions on large GPU EAM runs, use `eam_decomp_perf_gpu_heavy` for a single
   representative point and `eam_decomp_zone_sweep_gpu` for a manual layout sweep before changing
   any zoning policy.
7. If large-run TD speedup is smaller than expected, run `eam_td_breakdown_gpu` before changing
   kernels or zoning heuristics. It separates current runtime into `forces_full`,
   `target_local_force`, real device sync, cell-list build, candidate enumeration, and zone
   bookkeeping, and it records the current many-body force-scope contract explicitly. That keeps
   optimization work pointed at the actual dominant cost instead of assuming TD locality the
   runtime does not yet have.
   Use `baseline_reference_version=pr_mb01_v1` in the artifact to compare current CUDA behavior
   against the frozen pre-locality ceiling.
8. Once `eam_td_breakdown_gpu` confirms the corrected many-body locality model, use
   `td_autozoning_advisor_gpu` to turn current resource availability plus benchmark evidence into
   a recommendation-only TD zoning plan. This advisor must not mutate runtime policy implicitly;
   it is an operator decision aid, not an auto-applied scheduler.
9. For large single-GPU crack-like workloads, do not interpret "smaller `z` is faster" as proof
   that TD is the wrong method. Current representative evidence says:
   - TD still beats space decomposition at equal `z`,
   - the current one-zone-at-a-time `1D` slab implementation simply pays too much orchestration
     cost as `z` grows.
10. The next backend optimization track is therefore **single-GPU `1D` slab wavefront**
   execution:
   - batch several formally independent slabs in one GPU wave,
   - prefer fused launches and slab-local neighbor reuse over many tiny streams,
   - keep 3D decomposition as a correctness/general-purpose path rather than the first scaling
     target for this work.
11. Read current operator artifacts through the `PR-SW01` contract lens:
   `eam_decomp_zone_sweep_gpu`, `td_autozoning_advisor_gpu`, and `al_crack_100k_compare_gpu`
   now expose passive wavefront candidate fields before any fused execution path exists.

## Next Active GPU Track
- `PR-SW01..PR-SW05` define and validate single-GPU slab-wavefront execution.
- `PR-SW01` is complete:
  passive contract helper `tdmd/wavefront_1d.py` (`pr_sw01_v1`) now emits first-wave size,
  deferred-zone counts, and fallback-to-sequential reasons in the current operator benchmarks.
- `PR-SW02` is complete:
  `scripts/bench_slab_wavefront_evidence_gpu.py` and preset `slab_wavefront_evidence_gpu`
  now keep crack compare, crack `z=1..12` sweep, control sweep, and control breakdown as the
  first-class operator evidence surface for this track.
- `PR-SW03` is complete:
  `tdmd/wavefront_reference.py`, `scripts/bench_wavefront_reference_equivalence.py`, and preset
  `wavefront_reference_smoke` now prove grouped `1D` slab waves against the current sequential
  CPU slab semantics before any CUDA batching path exists.
- `PR-SW04` is complete:
  `tdmd/td_local.py` now exposes runtime contract `pr_sw04_v1`; admissible CUDA `1D` slab waves
  fuse the pre-force half-step across the wave while zone state progression remains
  sequential-per-zone and `W<=1` stays intact.
- Goal:
  allow several formally independent `1D` slab zones to share one GPU wave without hidden
  barriers and without changing TD-observable behavior.
- Next active step:
  `PR-SW05` profiling, strict acceptance, and advisor-cost integration on top of the proven
  `PR-SW03` reference semantics and current `PR-SW04` runtime contract.
  Current implementation already adds passive runtime diagnostics contract `pr_sw05_v1` to
  `tdmd/td_local.py` and propagates those realized cost-model fields into the operator
  `EAM/eam-alloy` artifacts and recommendation-only advisor.
- Non-goal:
  generic "concurrent arbitrary zones" execution. Independence must be explicit, proven, and
  instrumented before any batching lands.
- Go/no-go:
  keep this track only if representative wavefront execution produces a real wall-clock win on at
  least one large workload over the current sequential `1D` slab baseline.

## Historical Milestone (PR-G02)
- Added CUDA pair-force kernels for `LJ/Morse` in `tdmd/forces_gpu.py`.
- `tdmd/serial.py` and `tdmd/td_local.py` dispatch to GPU pair kernels when:
  - `device` resolves to `cuda`,
  - potential is pairwise (`LJ/Morse`).
- If CUDA backend is unavailable or potential is unsupported, code paths fall back to CPU force kernels.

## Historical Milestone (PR-G03)
- Added GPU cell-list candidate path in `tdmd/forces_gpu.py::forces_on_targets_celllist_backend`.
- Candidate sets are built with CPU-equivalent cell buckets and neighbor-stencil logic, then pair interactions are evaluated on GPU.
- Parity checks are covered by tests (`tests/test_gpu_pair_kernels.py`) with CUDA-enabled assertions when hardware/backend is available.

## Historical Milestone (PR-G04)
- Extended GPU force runtime to `table` potential in `tdmd/forces_gpu.py`.
- `serial` and `td_local` can execute table force evaluations through the same backend-dispatch path used for pair kernels.
- Current CUDA-cycle refinement: direct table target-force evaluation no longer relies on dense
  broadcast fallback; it uses the neighbor-list-driven RawKernel path.

## Historical Milestone (PR-G05)
- Added EAM/alloy CUDA force-evaluation path in `tdmd/forces_gpu.py` (`EAMAlloyPotential` branch).
- `serial` and `td_local` can dispatch EAM force evaluation to GPU backend (with CPU fallback when CUDA is unavailable).
- Current CUDA-cycle refinement: direct EAM/alloy target-force evaluation uses a device-side
  two-pass neighbor pipeline (`density -> dF -> force`) over the candidate-union neighbor list,
  removing the dense `cand x cand` path and Python-level element loops from the GPU branch.

## Historical Milestone (PR-G06)
- `td_local` host automaton pipeline is wired to use GPU force backend for supported potentials while keeping existing TD scheduling/state logic unchanged.
- Sync and async td_local parity checks are covered in `tests/test_gpu_pair_kernels.py` (CUDA-enabled checks where backend is available).

## Historical Milestone (PR-G07)
- Added local-rank to device mapping helpers:
  - `tdmd/backend.py::local_rank_from_env`
  - `tdmd/backend.py::cuda_device_for_local_rank`
- `tdmd/td_full_mpi.py` uses rank-local device mapping for `1 rank = 1 GPU` assignment when CUDA backend is selected.
- MPI smoke coverage includes 2-rank and 4-rank static_rr cases (`examples/td_1d_morse_static_rr.yaml`, `examples/td_1d_morse_static_rr_smoke4.yaml`).

## Historical Milestone (PR-G08)
- Added optional MPI transport refinements in `tdmd/td_full_mpi.py`:
  - `td.comm_overlap_isend`: nonblocking `Isend` path for payload delivery with batched `Waitall`.
  - `td.cuda_aware_mpi`: enables CUDA-aware transport mode when backend resolves to `cuda` (falls back to host-staging when inactive).
- Added overlap benchmarking helper:
  - `scripts/bench_mpi_overlap.py` compares blocking vs overlap path and writes
    `results/mpi_overlap.csv` and `results/mpi_overlap.md`.
- These options are transport-level refinements and do not alter TD automaton state transitions.

## Post-Baseline Hardening (PR-H01)
- Added hardware-strict GPU verification gate in `scripts/run_verifylab_matrix.py`:
  - backend evidence is persisted to `results/<run_id>/config.json` and `results/<run_id>/summary.json`,
  - when `require_effective_cuda` is enabled, CUDA request resolving to CPU fallback is treated as verification failure.
- Added hardware-strict GPU presets:
  - `gpu_smoke_hw`, `gpu_interop_smoke_hw`, `gpu_metal_smoke_hw`.
- This changes verification policy only and does not alter runtime TD semantics.

## Strict-Gate Plan (CUDA Cycle)
- Existing strict GPU gates remain mandatory throughout migration:
  - `gpu_smoke`, `gpu_interop_smoke`, `gpu_metal_smoke`,
  - `gpu_smoke_hw` (CUDA hardware-strict lane),
  - `gpu_perf_smoke` (CUDA transfer/kernel perf smoke lane).
- No-fallback acceptance rule remains mandatory for hardware-strict lanes.

## Post-Baseline Hardening (PR-H02)
- Added strict TD-MPI overlap A/B presets in `scripts/run_verifylab_matrix.py`:
  - `mpi_overlap_smoke`,
  - `mpi_overlap_cudaaware_smoke`.
- Presets invoke `scripts/bench_mpi_overlap.py` for ranks `2` and `4`, persist per-rank artifacts, and enforce strict invariant counters (`hG`, `hV`, `violW`, `lagV`).
- This is verification/transport-policy hardening and does not alter TD automaton semantics.

## Post-Baseline Hardening (PR-H03)
- Strict VerifyLab runs now enforce geometry guardrails (`strict_min_zone_width=True`) in TD-local verification calls.
- Task-based strict smoke presets use geometry-valid zone layouts to avoid warning-only acceptance.
- This is acceptance-policy hardening only; compute/automaton semantics are unchanged.

## Post-Baseline Hardening (PR-H06/H07)
- Long-horizon async drift governance is enforced by `longrun_envelope_ci` and baseline file `golden/longrun_envelope_v1.json`.
- Strict VerifyLab failures auto-produce incident bundles (`tdmd/incident_bundle.py`) and can be manually exported via `scripts/export_incident_bundle.py`.
- These are verification/triage policy layers only; runtime TD semantics and backend execution semantics are unchanged.

## Risk Burndown v2 (PR-V2-01..PR-V2-04)
- Added profile-driven cluster validation contract (`tdmd/cluster_profile.py`, `examples/cluster/*.yaml`).
- Added strict transport matrix benchmarking (`scripts/bench_mpi_transport_matrix.py`) and VerifyLab preset `mpi_transport_matrix_smoke`.
- Added strict cluster scaling/stability lanes:
  - `cluster_scale_smoke`,
  - `cluster_stability_smoke`.
- These lanes validate overlap/cuda-aware transport behavior under profile-controlled conditions and do not alter TD automaton or backend semantics.

## CUDA Cycle Progress (PR-C05)
- Persistent device-state caching now keeps positions/types resident on GPU across repeated force evaluations for the same host state object.
- Runtime paths in `serial` and `td_local` explicitly mark updated host atoms as dirty after position-changing steps; the next CUDA force call syncs only the marked subset instead of re-uploading the full target/candidate union.
- Static potential tables are cached on device for:
  - `table` (`r_grid`, `f_grid`),
  - `eam/alloy` (`grid_r`, `grid_rho`, `density`, `embed`, `phi`).
- Direct external GPU force API semantics are preserved:
  - default calls still perform conservative state syncs,
  - incremental dirty-range sync is enabled only for runtime-managed paths that explicitly opt into it.

## CUDA Cycle Progress (PR-C06)
- Added TD-MPI compute-path GPU refinement hook in `tdmd/td_full_mpi.py`:
  - work-zone force callback uses GPU force dispatch when backend is CUDA,
  - automatic fallback to wrapped CPU potential keeps reference semantics.
- TD-MPI runtime-managed CUDA path now uses the same persistent device-state policy as `serial` and `td_local`:
  - GPU refinement force dispatch opts into `prefer_marked_dirty=True`,
  - work-zone compute marks pre-migration atom ids dirty after Velocity-Verlet position updates,
  - MPI recv path marks received atom ids dirty after host-staging updates,
  - ensemble box rescale marks the device state dirty only when positions change.
- This integration does not change TD automaton transitions or communication ordering.
- Added TD-MPI overlap observability counters in runtime diagnostics:
  - `sendPackMs`, `sendWaitMs`, `recvPollMs`, `overlapWinMs`.
  These counters are consumed by overlap VerifyLab lanes as observability-only metrics.
- PR-C06 acceptance keeps overlap/cuda-aware/cluster transport strict lanes green without introducing
  new implicit barriers.

## CUDA Cycle Progress (PR-C07)
- RawKernel handles are now cached per backend/device in `tdmd/forces_gpu.py` to reduce repeated
  Python-side kernel construction overhead without changing dispatch semantics.
- `scripts/profile_gpu_backend.py` now consolidates:
  - verify-preset CPU-vs-GPU timing ratios,
  - `gpu_perf_smoke` transfer/kernel metrics,
  - current `EAM/alloy` decomposition speedups,
  - optional `Phase E` comparison against commit `efb864e` via `--phase-e-worktree`.
- Added `scripts/bench_eam_td_breakdown_gpu.py` and manual preset `eam_td_breakdown_gpu` for
  profiling why large single-GPU `EAM/eam-alloy` TD speedups may remain modest even when CUDA is
  effective; the artifact now records the explicit many-body force-scope baseline
  (`pr_mb01_v1`) alongside the timing breakdown and current contract version.
- Current PR-C07 decision point result:
  stay on CuPy RawKernel.
  Representative `EAM/alloy` profiling (`512 atoms`, `2 steps`) shows current CUDA path beating
  both CPU reference and the `Phase E` baseline commit `efb864e`; `C++/CUDA` extension remains
  optional Plan B, not the active path.
