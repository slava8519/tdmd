# TD Automaton Specification

This spec reflects the current implementation in `tdmd/td_automaton.py` and
`tdmd/td_full_mpi.py`.
Mode-level guarantees and verification gate ownership are summarized in `docs/MODE_CONTRACTS.md`.

## States (per zone)
- **F**: empty / no local state (`ZoneType.F`)
- **D**: data-ready (owned or shadow data present)
- **P**: locked donor (dependency held while another zone computes)
- **W**: working (zone currently being computed)
- **S**: computed, pending send / publish

## Events
- **RECV_ZONE**: receive full zone state (startup or `REC_ZONE`)
- **RECV_DELTA_MIGRATION**: receive migrated atoms
- **RECV_DELTA_HALO**: receive halo (ghost) atoms
- **START_COMPUTE**: select a D zone to compute
- **FINISH_COMPUTE**: compute step completed
- **SEND_ZONE**: publish zone to owner / next holder (or local finalize in `static_rr`)

## Transition Table (core)

| From | Event | Preconditions | To | Postconditions | Code |
|---|---|---|---|---|---|
| F | RECV_ZONE | zone payload received | D | `atom_ids` set, `step_id` updated | `td_full_mpi.startup_distribute_zones`, `td_automaton.on_recv` |
| D | START_COMPUTE | deps_table ok, deps_owner ok, lag ok, no other W | W | donors locked as P, table built | `TDAutomaton1W.start_compute` |
| D | START_COMPUTE (donor) | chosen as dependency for W zone | P | donor locked | `TDAutomaton1W.start_compute` |
| W | FINISH_COMPUTE | step forces computed | S | `step_id` advanced, `halo_ids` cleared, send queued | `TDAutomaton1W.compute_step_for_work_zone` |
| P | FINISH_COMPUTE (release) | W zone finished | D | donor unlocked | `TDAutomaton1W.compute_step_for_work_zone` |
| S | SEND_ZONE | sender publishes zone (dynamic/static_3d) | F | local state cleared if ownership moves | `td_full_mpi` send phase |
| S | SEND_ZONE | `static_rr` ownership (no REC_ZONE) | D | local zone remains owned; only halos/deltas routed | `td_full_mpi` send phase |

## Delta / Halo Updates (no state change)
- **RECV_DELTA_MIGRATION** updates `atom_ids` for the target zone-time.
- **RECV_DELTA_HALO** updates `halo_ids` (single-step layer).
Code: `td_full_mpi` delta handling (migration + halo), `ZoneRuntime.halo_ids`.

## Preconditions Summary
- **deps_table**: all table dependencies present locally (owned or shadow)
- **deps_owner**: owner deps known and fresh (if enabled)
- **lag**: dependencies not older than `max_step_lag`
Code: `_deps_table_ok`, `_deps_owner_ok`, `_lag_ok` in `td_automaton.py`.

## Notes
- The **formal core** invariant is enforced: at most one zone in W on a rank.
- `sync_mode` exists only in `td_local.py` for verification and does not alter TD semantics.
- In **`static_rr`** mode, ownership is fixed (`owner_rank(z)=zid % P`): `REC_ZONE` is only used at startup,
  and `SEND_ZONE` does not transfer ownership. Only `HALO` and `MIGRATION` deltas are routed.
- Verification gating (strict vs diagnostic presets) is defined in `docs/VERIFYLAB.md` and does not change this automaton.
- Task run-compatibility checks (`tdmd/io/task.py::validate_task_for_run`) are pre-run input validation and do not change automaton transitions.
- Task/config ensemble checks (`nve|nvt|npt`) and mode-level runtime guardrails (`td_full_mpi` NPT single-rank constraint) are pre-run/runtime policy checks and do not change automaton transitions.
- Ensemble controllers (`NVT/NPT` in `tdmd/ensembles.py`) apply post-step velocity/box scaling only; they do not add automaton states/events/transitions.
- Pair-runtime mass/type support, `lj`/`morse` type-dependent pair matrices (`potential.params.pair_coeffs`), `potential.kind=table` runtime, and EAM backends (`serial` + `td_local` + `td_full_mpi` compute callback) are implemented outside the automaton layer.
- TD-MPI task path now accepts per-atom mass arrays (non-uniform alloys); this changes numeric integration inputs only and does not alter automaton states/events/transitions.
- Materials parity suite checks (`scripts/materials_parity_pack.py`) are verification/acceptance policy and do not introduce new automaton events or transitions.
- Long-run envelope verification (`longrun_envelope_ci`, baseline `golden/longrun_envelope_v1.json`) is acceptance policy only and does not alter automaton states/events/transitions.
- Incident-bundle generation (`tdmd/incident_bundle.py`, `scripts/export_incident_bundle.py`) is diagnostics/triage policy only and does not alter automaton states/events/transitions.
- Time-blocking (`td.time_block_k` / `td.batch_size`) affects only send-queue dequeue granularity (`pop_send_batch`) and message batching cadence; it does not add transitions, does not change F/D/P/W/S semantics, and does not introduce global barriers.
- Compute backend selection (`run.device`) is outside automaton semantics; it selects execution backend only and does not modify transitions/events.
  - current implementation: `auto|cpu|cuda`,
  - planned portability extension: `auto|cpu|cuda|hip|kokkos` (see `docs/PORTABILITY_KOKKOS_PLAN.md`).
- GPU pair-force kernels for `LJ/Morse` (`serial`/`td_local`) are compute refinements only; automaton state transitions/messages remain unchanged.
- MPI overlap transport flags (`td.comm_overlap_isend`, `td.cuda_aware_mpi`) change message transport strategy only (blocking vs nonblocking/cuda-aware path) and do not introduce new automaton states or transitions.
- Hardware-strict VerifyLab backend gate (`require_effective_cuda` currently; planned `require_effective_gpu`) affects acceptance policy only and does not alter automaton states/events/transitions.
- MPI overlap strict VerifyLab presets (`mpi_overlap_smoke`, `mpi_overlap_cudaaware_smoke`) and benchmark invariant checks (`hG/hV/violW/lagV`) are verification-policy checks only and do not alter automaton transitions.
- Strict guardrail enforcement (`strict_min_zone_width=True` under `--strict` VerifyLab runs) is input/geometry validation policy and does not alter automaton transitions.
- Cluster validation scripts (`bench_cluster_scale.py`, `bench_cluster_stability.py`, `bench_mpi_transport_matrix.py`) are profile-driven verification policy only and do not introduce new automaton events/transitions.
- Materials parity suite v2 / property-level gates (`materials_parity_suite_v2`, `materials_property_gate.py`) are acceptance policy only and do not alter F/D/P/W/S semantics.

## Mode-Specific Messaging
**Ring-transfer (core):**
- Uses `REC_ZONE` in steady-state to transfer ownership along the ring.
- `SEND_ZONE` transitions `S -> F` locally (ownership moves), then the receiver enters `D`.

**Static round-robin (`static_rr` extension):**
- Uses `REC_ZONE` only at startup (fixed ownership).
- `SEND_ZONE` transitions `S -> D` locally (no ownership transfer).
- Only `REC_DELTA_MIGRATION` and `REC_DELTA_HALO` are routed in steady-state.
