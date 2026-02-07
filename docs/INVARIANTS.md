# Invariants

This file lists key TD invariants and where they are checked.
Mode-level guarantees and strict-gate policy mapping are documented in `docs/MODE_CONTRACTS.md`.
VerifyLab strict/diagnostic preset policy is documented in `docs/VERIFYLAB.md`; it does not alter invariant definitions below.
Task run-compatibility validation (`tdmd/io/task.py::validate_task_for_run`) is an input contract check and does not redefine these invariants.
Task/config ensemble validation (`nve|nvt|npt`) and mode-level runtime guardrails (including `td_full_mpi` NPT single-rank constraint) are input/runtime-policy checks only; they do not redefine invariant predicates/counters.
Pair-runtime multi-mass support, type-dependent pair-coefficient matrices, `table` potential runtime, and EAM backends (`serial`/`td_local`/`td_full_mpi` force callback) affect force/integration numerics only; invariant definitions remain unchanged.
Task-driven TD-MPI support for non-uniform per-atom masses (alloy path) changes only integration inputs and does not redefine invariant predicates/counters.
LAMMPS interop/export and VerifyLab materials presets are verification/IO features and do not alter automaton invariants.
Materials parity suite checks (`scripts/materials_parity_pack.py`, fixture `examples/interop/materials_parity_suite_v1.json`) consume runtime observables/metrics and do not redefine invariant predicates/counters.
Long-run envelope baseline checks (`longrun_envelope_ci`, `golden/longrun_envelope_v1.json`) consume existing drift metrics (`final_*`, `rms_*`) and do not redefine invariant predicates/counters.
Incident bundle generation/exports (`tdmd/incident_bundle.py`, `scripts/export_incident_bundle.py`) consume run artifacts and diagnostics only; invariant predicates/counters are unchanged.
Time-blocking (`td.time_block_k` / `td.batch_size`) changes only send-batch granularity and adds diagnostics (`send_batches`, `send_batch_zones_total`, `send_batch_size_max`); it does not introduce a new semantic invariant.
Backend selection (`run.device`) is execution-path selection only; invariants remain defined over TD states and dependency/geometry predicates.
GPU pair-force kernels (`LJ/Morse` in `serial`/`td_local`) keep the same target/candidate interaction sets as CPU and do not redefine invariant checks.
GPU table/EAM force paths follow the same rule: backend changes numeric execution path only, not TD state invariants.
MPI overlap transport options (`td.comm_overlap_isend`, `td.cuda_aware_mpi`) affect message delivery mechanics only and do not redefine invariant predicates or counters.
Hardware-strict GPU verification gate (`require_effective_cuda` currently; planned `require_effective_gpu`) is an acceptance-policy check and does not redefine invariant predicates/counters.
MPI overlap strict presets and benchmark parsing of `hG/hV/violW/lagV` provide additional verification gates; they consume existing counters and do not redefine invariants.
Strict VerifyLab guardrails (`strict_min_zone_width` enabled under `--strict`) enforce geometry preconditions and do not redefine invariant predicates/counters.
Cluster profile-driven validation lanes (`cluster_scale_smoke`, `cluster_stability_smoke`, `mpi_transport_matrix_smoke`) consume existing counters/artifacts only and do not redefine invariant predicates/counters.
Materials parity suite v2 and property-level strict gates (`materials_property_gate.py`) consume existing observables/drift metrics only and do not redefine invariant predicates/counters.

## I0. Formal core: at most one W per rank
- **Meaning**: a rank cannot compute two zones simultaneously.
- **Check**: `TDAutomaton1W._assert_invariants` raises and increments `viol_w_gt1`.
- **Code**: `tdmd/td_automaton.py`.

## I1. Table-deps readiness
- **Meaning**: a zone may enter W only if all `deps_table(z)` are locally represented.
- **Check**: `wait_table` counter and `_deps_table_ok` gating.
- **Code**: `tdmd/td_automaton.py`, `tdmd/td_full_mpi.py` (predicates).

## I2. Owner-deps readiness (optional)
- **Meaning**: if enabled, all `deps_owner(z)` must have a known, fresh holder.
- **Check**: `wait_owner`, `wait_owner_unknown`, `wait_owner_stale`.
- **Code**: `tdmd/td_full_mpi.py`.

## I3. Halo ⊆ support(table(z)) (3D)
- **Meaning**: every halo atom must be inside table support.
- **Check**: `hV3` counter (3D), `hV` in MPI logs (1D).
- **Code**: `tdmd/td_automaton.py` (`hV3`), `tdmd/td_full_mpi.py` (`hV`).

## I4. Halo geometry (p-neighborhood)
- **Meaning**: halo atoms must lie within zone AABB expanded by cutoff.
- **Check**: `hG3` counter (3D), `hG` in MPI logs (1D).
- **Code**: `tdmd/td_automaton.py` (`hG3`), `tdmd/td_full_mpi.py` (`hG`).

## I5. Table-geometry support (3D)
- **Meaning**: candidate ids used for table must be inside AABB support.
- **Check**: `tG3` counter.
- **Code**: `tdmd/td_automaton.py::ensure_table`.

## I6. Time-lag safety
- **Meaning**: dependencies cannot lag more than `max_step_lag`.
- **Check**: `_lag_ok` gating and `viol_lag` counter.
- **Code**: `tdmd/td_automaton.py`.

## I7. Buffer sufficiency (diagnostic)
- **Meaning**: buffer covers drift implied by `max_step_lag`.
- **Check**: `viol_buffer` counter.
- **Code**: `tdmd/td_full_mpi.py::update_buffers`, `tdmd/zones.py::compute_zone_buffer_skin`.
