## 4.5.6
- Replaced active GPU planning strategy with CUDA-only execution cycle (`PR-C01..PR-C08`).
- Added `docs/CUDA_EXECUTION_PLAN.md` (numba-cuda primary; plan B RawKernel/C++ CUDA extension).
- Archived `docs/PORTABILITY_KOKKOS_PLAN.md` (historical only).
- Aligned governance/docs/prompts (`AGENTS.md`, `docs/TODO.md`, `docs/ROADMAP_GPU.md`,
  `docs/PR_PLAN_GPU.md`, `docs/MODE_CONTRACTS.md`, `docs/VERIFYLAB.md`,
  `CODEX_MASTER_PROMPT.md`, `CODEX_GPU_MASTER_PROMPT.md`, `README.md`).

## 4.5.5
- Added `scripts/bootstrap_codex_env.sh` for one-command Codex workstation bootstrap
  (system tools + `.venv` + editable install with dev extras + shell aliases).
- Added `CODEX_ENV_BOOTSTRAP_PROMPT.md` for fresh-machine Codex startup.
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
