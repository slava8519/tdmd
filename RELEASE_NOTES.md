## 4.5.22
- Added first-class Al microcrack operator benchmark scripts:
  `scripts/generate_al_crack_task.py` and `scripts/bench_al_crack_compare.py`.
- Added VerifyLab preset `al_crack_100k_compare_gpu` plus `make al-crack-100k-gpu` for a
  `100K`-atom pure-Al `eam/alloy` GPU comparison with requested `1000` zones, exact-request TD
  preflight reporting, strict-valid fallback comparison, and per-case telemetry artifacts.
- VerifyLab markdown/stdout now prints the benchmark report for this mode, including observed step
  counts and telemetry sidecar locations after timeout or completion.

## 4.5.21
- Added optional runtime telemetry JSONL output for `tdmd.main run` across `serial`, `td_local`,
  and `td_full_mpi`.
- Telemetry records step progress, ETA, model-state summaries, process resource usage, and CUDA
  device/pool memory fields when the effective device is `cuda`.
- Added `--telemetry`, `--telemetry-every`, `--telemetry-heartbeat-sec`, and
  `--telemetry-stdout` CLI controls plus manifest/summary sidecars for long-running operator
  benchmarks.

## 4.5.20
- Completed `PR-ML02`: added strict VerifyLab and fixture-driven acceptance for the
  `quadratic_density` `ml/reference` family.
- Added task-based strict preset `ml_reference_smoke`, interop tasks under `examples/interop/`,
  and versioned fixture suite `examples/interop/ml_reference_suite_v1.json`.
- Added `scripts/ml_reference_parity_pack.py --strict` and explicit LAMMPS export rejection tests
  for `ml/reference`, so the current interop boundary is clear rather than implicit.

## 4.5.19
- Completed `PR-ML01`: added `potential.kind=ml/reference` as a versioned CPU-reference many-body
  contract for future ML-potential work.
- The new contract explicitly validates `cutoff`, `descriptor`, `neighbor`, and `inference`
  fields, rejects hidden global barriers, and requires `task.cutoff >= contract.cutoff.radius`.
- Added CPU reference harness coverage for `serial`, `td_local`, task-driven CLI runtime, and
  task-based VerifyLab sweep path, while intentionally leaving fixture-driven ML strict presets and
  interop coverage for `PR-ML02`.

## 4.5.18
- Completed `PR-ZA01`: added `scripts/bench_td_autozoning_advisor.py` and manual VerifyLab preset
  `td_autozoning_advisor_gpu` for resource-aware TD zoning recommendations on `EAM/eam-alloy`.
- The advisor detects visible CPU/GPU/MPI resources, enumerates strict-valid layouts, benchmarks
  paired `space_gpu` vs `time_gpu`, and emits recommendation-only markdown/json/csv artifacts.
- Added optional `pr_za01_v1` breakdown evidence for the recommended layout, referencing the
  corrected `PR-MB03` many-body locality baseline instead of mutating runtime zoning policy.

## 4.5.17
- Completed `PR-MB03`: CUDA async `td_local` many-body now uses target/candidate-local GPU
  dispatch first, with full-system GPU fallback only if that refinement path is unavailable.
- Updated `eam_td_breakdown_gpu` observability so current runs report contract version
  `pr_mb03_v1` and reference the frozen pre-locality baseline `pr_mb01_v1`.
- On the representative `10K` `EAM/eam-alloy` benchmark, current GPU runs now report
  `target_local_force_calls=3072` and `forces_full_share=0` for both `space_gpu` and `time_gpu`.

## 4.5.16
- Completed `PR-MB02`: CPU async `td_local` many-body now uses target-local
  `potential.forces_on_targets(...)` instead of repeated full-system
  `forces_full(ctx.r)[ids0]`.
- Added regression coverage proving the new CPU target-local path matches the legacy full-system
  behavior for representative async `1d` and `3d` `EAM/eam-alloy` cases.
- Kept the GPU many-body baseline unchanged for now; `eam_td_breakdown_gpu` should still report
  the pre-`PR-MB03` full-system CUDA force scope.

## 4.5.15
- Added explicit many-body force-scope contract helpers in `tdmd/many_body_scope.py` and
  `tdmd/td_local.py::describe_many_body_force_scope`.
- Froze `eam_td_breakdown_gpu` baseline contract as `pr_mb01_v1`, including explicit
  `evaluation_scope`, `consumption_scope`, `target_local_available`, and
  `target_local_force_calls` observability fields.
- Marked the next algorithmic step as `PR-MB02`: replace the current repeated full-system
  many-body `td_local` path with a CPU-reference target-local implementation before further
  zoning or ML-potential policy work.

## 4.5.14
- Added `scripts/bench_eam_td_breakdown_gpu.py` and manual preset `eam_td_breakdown_gpu` for
  GPU-only `10K`-atom `EAM/eam-alloy` runtime attribution.
- Added operator target `make eam-td-breakdown-gpu`.
- Documented the breakdown workflow as the first profiling step before changing TD zoning policy
  or claiming the current TD optimization ceiling.

## 4.5.12
- Stabilized `gpu_perf_smoke` methodology with a calibrated synthetic kernel timing floor.
- Surfaced calibration metadata in `gpu_perf_smoke` summaries and `gpu-profile` markdown.
- Kept `gpu_perf_smoke` thresholds unchanged while reducing timer-jitter sensitivity.
- Added manual `eam_decomp_perf_gpu_heavy` benchmark for GPU-only `10K`-atom `EAM/eam-alloy`
  TD-vs-space performance comparison outside the PR smoke lane.

## 4.5.13
- Added `scripts/bench_eam_zone_sweep_gpu.py` and manual preset `eam_decomp_zone_sweep_gpu`
  for GPU-only `10K`-atom `EAM/eam-alloy` TD-vs-space zone-layout sweeps.
- Added operator entrypoint `make eam-zone-sweep-gpu`.
- Planned future resource-aware TD auto-zoning as an observability/recommendation layer in `docs/TODO.md`.

## 4.5.11
- Added PR-C08 docs/ops consolidation:
  - marked CUDA execution cycle `PR-C01..PR-C08` complete in roadmap/plan docs,
  - synchronized prompts/governance around the current `stay_on_rawkernel` decision,
  - added operator playbook guidance for `gpu-profile`, representative `EAM/alloy` workload,
    and optional `Phase E` comparison against commit `efb864e`.

## 4.5.10
- Added PR-C07 incremental profiling/runtime hardening:
  - cached `CuPy RawKernel` handles per backend/device in `tdmd/forces_gpu.py`,
  - upgraded `scripts/profile_gpu_backend.py` into a consolidated CUDA-cycle profiler with
    markdown/json output, `gpu_perf_smoke`, and `EAM/alloy` benchmark integration,
  - documented optional `Phase E` comparison against commit `efb864e`.

## 4.5.9
- Added PR-C06 TD-MPI CUDA refinement:
  - `td_full_mpi` GPU force dispatch now opts into runtime-managed dirty-range syncing,
  - work-zone compute and MPI receive paths mark only changed atom ids dirty,
  - NPT box rescale marks device state dirty only when positions change.
- Kept overlap/cuda-aware transport semantics and observability counters unchanged while passing the
  overlap/cluster/transport strict matrix.

## 4.5.8
- Added PR-C05 CUDA runtime refinement:
  - persistent device-state caching for positions/types in runtime-managed GPU paths,
  - dirty-range host->device sync after host-side position updates,
  - device caching for `table` and `eam/alloy` potential tables.

## 4.5.7
- Added `scripts/bench_eam_decomp_perf.py` for standard `EAM/alloy` comparison across
  `space_cpu`, `space_gpu`, `time_cpu`, and `time_gpu`.
- Added VerifyLab preset `eam_decomp_perf_smoke`; `make verify-smoke` and PR CI now run it
  and print the benchmark table after completion.

## 4.5.6
- Replaced active GPU planning strategy with CUDA-only execution cycle (`PR-C01..PR-C08`).
- Added `docs/CUDA_EXECUTION_PLAN.md` (`CuPy RawKernel` primary; `C++/CUDA` extension as Plan B).
- Aligned governance/docs/prompts (`AGENTS.md`, `docs/TODO.md`,
  `docs/MODE_CONTRACTS.md`, `docs/VERIFYLAB.md`,
  `CODEX_MASTER_PROMPT.md`, `CODEX_GPU_MASTER_PROMPT.md`, `README.md`).

## 4.5.5
- Added `scripts/bootstrap_codex_env.sh` for one-command Codex workstation bootstrap
  (system tools + `.venv` + editable install with dev extras + shell aliases).
- Added `pyproject.toml` packaging metadata for editable installs (`pip install -e .`)
  with explicit package discovery (`tdmd*`) and project dependencies.

## 4.5.4
- Added docs/CONTRACTS.md with hard do-not-break rules for Codex cycles.
- Added regression tripwire tests for W<=1 enforcement and A4b tie-break presence.

## 4.5.3
- Added PR-0 golden SoA data contract (docs/PR0_DATA_CONTRACT.md) and tdmd/zone_views.py builders.
- Added unit test for zone view contiguity/dtypes.

## 4.5.1
- Added docs/GPU_BACKEND_API.md: strict contract for GPU compute backend integration.

## 4.5.0
- Docs/structure cleanup + GPU planning docs.
- Unified versioning via root VERSION file.
- Added GPU Backend Engineer agent + CODEX GPU master prompt.

## 4.4.8
- VerifyLab mpi_overlap summary: Pareto frontier (speedup vs wfgC_rate) + suggested operating points under cycle-rate budgets.

## 4.4.7
- Normalized WFG metrics exported: wfgC_rate and wfgC_per_100_steps (mpi_overlap CSV + verifylab summary).

## 4.4.6
- Scientific WFG diagnostics summary: reports max(wfgC), max(wfgO), cycle_rate and ranks where worst contention observed.

## 4.4.5
- VerifyLab summary emits non-fatal warning if transient local WFG cycles are observed in mpi_overlap mode.

## 4.4.4
- Export WFG diagnostics (wfgS/wfgC/wfgO) into MPI overlap bench CSV and verifylab summary.

## 4.4.2\n- A4b runtime cycle-breaking via priority scheduling (step_id, owner_rank, zone_id).\n- progress_epoch instrumentation for liveness.\n- chaos no-stall liveness test (single-rank).\n\n# Release Notes

This file is the project changelog. `README.md` documents the project **as it is today**.

GitHub releases:
- https://github.com/slava8519/tdmd/releases

## v0.1.0 (2026-02-07)

Initial public bootstrap release.

Highlights:
- repository publication readiness for GitHub,
- strict CI smoke alignment (`smoke_ci --strict`, `interop_smoke --strict`),
- GPU portability cycle plan (Kokkos, NVIDIA+AMD) captured as PR plan (`PR-K01..PR-K10`),
- repository portability cleanup (no machine-specific absolute paths in tracked docs/fixtures).

Tag:
- `v0.1.0`

## Legacy TDMD-TD Milestone Notes (pre-release numbering)

These notes were historically tracked in `README.md` and are preserved here for reference.

### v1.9 (added) — buffer/skin ↔ time-lag
- Buffer/skin scaled with `max_step_lag`: `buffer ~ vmax*dt*(L+1)`.
- Added `bufV` log counter for insufficient buffer coefficient vs linear drift estimate.

### v2.0 (added) — unified records wire format
- Unified message protocol: computed zones + outbox atoms travel in one stream via records:
  - `REC_ZONE` full zone state (zid, ids, r, v, step_id)
  - `REC_DELTA` zone delta (zid, ids, r, v), without changing zone step_id
- Removed extra tags/channels for outbox; single `tag_base`.

### v2.1 (added) — strict DELTA time targeting
- `REC_DELTA` carries target `step_id` (zone-time).
- Receiver policy:
  - apply if zone already at step,
  - defer until zone reaches step,
  - drop if too old (`zone.step_id - delta.step_id > max_step_lag`) → `dX`.
- Added log counters: `dA` applied, `dD` deferred, `dX` dropped.

### v2.2 (added) — DELTA subtypes + bounded pending buffer
- Added `subtype` field (`DELTA_MIGRATION=0` now, format ready for HALO/CORRECTION).
- `pending_deltas` bounded:
  - drops too-old groups by lag rule,
  - caps total deferred atoms via `td.max_pending_delta_atoms` (drops oldest by `step_id`).
- Added `dO` (dropped by overflow).

### v2.3 (added) — HALO as DELTA subtype (support(table)-driven)
- Added `DELTA_HALO=1`.
- When sending computed zone `z`:
  - atoms not transferable as ownership (`A(z) ∩ support(table(next))`) sent as `REC_DELTA(DELTA_HALO, ...)`,
  - ownership stays at sender; receiver uses halo for table/forces.
- Receiver stores HALO separately (`zone.halo_ids`), used only as candidates for `ensure_table`.
- Added `hA/hD/hX` counters.

### v2.4 (added) — single-layer HALO + uniquing
- Added `halo_step_id`.
- HALO becomes single-layer (replaces per step_id), `np.unique` applied.
- `ensure_table` uses halo only when `halo_step_id == step_id`.

### v2.5 (added) — geometric HALO filter + invariant
- Sender filters HALO to receiver p-neighborhood `[z0-cutoff, z1+cutoff]` with PBC.
- Receiver invariant: halo atoms must be inside p-neighborhood. Violations counted in `hG`.

### v2.6 (added) — HALO as support-driven minimal set
- HALO defined as `A(sender) ∩ support(table(receiver))` (not “all keep_ids”).
- Still filtered by receiver p-neighborhood.
- Added `hS` (sent halo atoms), `hV` (halo ⊆ support(table(zone)) violations).

### v2.7 (added) — bidirectional HALO
- Forward HALO stream (as before) to `next_rank` on `tag_base`.
- Backward HALO stream to `prev_rank` on `tag_base+2`.
- Added counters `hF` (forward), `hB` (backward).

### v2.8 (added) — HALO for full deps(z) list
- HALO formed for all dependency zones within cutoff-expanded interval.
- For each dep:
  - prefer support-driven overlap set when available,
  - fallback to geometric halo + receiver p-filter.
- Routed via two streams: forward to `next_rank`, backward to `prev_rank` (`tag_base+2`).

### v2.9 (added) — dynamic holder routing
- Introduced `holder_map[zid] -> rank` for where a zone lives right now.
  - init by allgather at startup,
  - updated on `REC_ZONE` send/recv.
- HALO and migration deltas routed directly to current holder rank.
- Receiver loop listens on `MPI.ANY_SOURCE` for `tag_base`.

### v3.0 (added) — deps readiness + holder gossip versioning
- Added `REC_HOLDER` update record (zid + version in `step_id` field).
- Introduced:
  - `holder_ver[zid]`,
  - local `holder_epoch`.
- Formal readiness:
  - `td.require_local_deps=true` (legacy default): zone enters W only if deps are local.
- `td.holder_gossip=true`: piggyback holder updates on zone transfers.

### v3.1 (added) — REQ_HALO pull protocol + shadow deps
- Added `REC_REQ` and `REQ_HALO=0` to request HALO for a dep zone.
- Holder replies with `DELTA_HALO(dep_zid)` directly to requester.
- On receiving HALO for a zone that was `F`, promote to shadow dep (`P`) with message step_id.
- `td.require_local_deps` reinterpreted: deps must be locally represented (owned or shadow/halo).
- Added proactive pull; counters `reqS/reqR`, `sh`.

### v3.2 (added) — table-deps vs owner-deps
- Two dependency classes:
  - table-deps for table/forces (halo/shadow is sufficient),
  - owner-deps for stronger ownership invariants (optional).
- New flags:
  - `td.require_table_deps` (default true),
  - `td.require_owner_deps` (default false),
  - `td.require_local_deps` kept as legacy alias for `require_table_deps`.
- Added counters `wT`, `wO`.

### v3.3 (added) — owner-deps via migration radius (buffer)
- owner-deps defined as zones intersecting `[z0-buffer, z1+buffer]` (with PBC).
- `td.require_owner_deps` checks holder is known (`holder_map[dep] != -1`).
- Added `wOU`.

### v3.4 (added) — REQ_HOLDER pull + freshness requirement
- Added `REQ_HOLDER=1` to request holder-map freshness.
- Freshness requirement (when `td.require_owner_ver=true`):
  - `holder_ver[dep] >= holder_epoch - (2*max_step_lag + 2)`.
- Proactive pull sends `REQ_HOLDER` when unknown/stale; counters `reqHS/reqHR`, `wOS`.

### v4.1 (added) — static_3d integration
- `static_3d` integrated in TD-MPI: deps over 3D blocks + routed ownership.
- HALO/overlap filtering for `static_3d` uses 3D AABB with PBC.

### v4.2 (added) — 3D CellList tables in W
- `static_3d`: interaction tables in W may build local 3D CellList (`InteractionTableState.impl = CellList`).
- `TDAutomaton1W.ensure_table` supports `geom_aabb` and/or `geom_provider`.
- TD-MPI `overlap_mode=geometric_rc` for `static_3d` uses AABB overlap.

### v4.3 (added) — 3D support AABB + invariants
- `InteractionTableState` stores 3D AABB support (`lo/hi`) for `static_3d`.
- Added 3D invariants: `hG3` (halo geo), `hV3` (halo in support), `tG3` (table candidate geo).
- Introduced `geom_pbc.py` for AABB checks under periodic boundaries.
# Unreleased

## Docs / Governance
- Refreshed post-CUDA roadmap to make the next active GPU direction explicit:
  single-GPU `1D` slab-wavefront TD scaling for large metals/alloys workloads.
- Recorded the current interpretation of the large-run Al microcrack evidence:
  TD still beats space decomposition at equal zone count on one GPU, but current TD runtime
  slows down as `z` increases because per-zone orchestration overhead grows faster than the
  current single-GPU implementation can amortize.
- Added a concrete PR queue for wavefront batching/fused execution and synchronized the
  GPU/operator prompt with that priority.
