# Liveness for TD deps-graphs

This document states **liveness (progress) lemmas** for TDMD’s dependency-graph execution.
Safety (no missing interactions) is handled separately in `docs/INVARIANTS.md`.

## 1. Model

- Zones `Z` evolve in discrete MD steps `t ∈ ℕ`.
- Each zone is controlled by the TD automaton (states F/D/P/W/S; see `docs/SPEC_TD_AUTOMATON.md`).
- Dependencies are represented by two directed graphs:
  - **owner-deps**: who must provide ownership/atom deltas for zone progress.
  - **table-deps**: who must provide halo/support data for interaction tables.

At a fixed step `t`, the runtime induces a **wait-for graph** `WFG(t)`:
an edge `a → b` means “zone a cannot progress at step t until some condition provided by zone b holds”
(e.g., owner-ready or table-ready).

## 2. Assumptions (A1–A4)

A1. **Reliable communication**  
All sent messages are delivered within finite time (no loss).

A2. **Weak fairness**  
If an action remains enabled indefinitely (message handling, scheduled transition), it eventually executes.

A3. **Bounded time-lag**  
The runtime enforces a bounded lag `L` between interacting zones (see `max_step_lag` and related guards).

A4. **No cyclic waiting at fixed step (or cycle breaking)**  
Either:
- (A4a) `WFG(t)` is acyclic for all `t` under the allowed deps-graph + protocol; or
- (A4b) the protocol implements a deterministic cycle-breaking rule (global priority/tie-break).

`dynamic` (dissertation core) satisfies A4 constructively via the ring ordering.
General deps-graphs MUST satisfy (A4a) or implement (A4b) explicitly.

## 3. Lemmas

### Lemma L0 (Monotone step)
For every zone `z`, `step(z)` is monotone non-decreasing.

### Lemma L1 (Delivery-to-enable)
Let a zone `z` wait for a finite set of messages/conditions at `(z,t)`.
Under A1 and A2, if all required sends occur, `z` becomes enabled in finite time.

### Lemma L2 (Local progress under enabledness)
If `z` is enabled at `(z,t)` and A2 holds, then `z` enters `W` and completes the compute transition in finite time.

### Lemma L3 (No deadlock under acyclic wait-for)
Assume A4a. Then for any fixed `t`, the system cannot deadlock at step `t`:
there exists at least one zone with no outgoing wait edges that can progress,
and by A1–A2 progress occurs in finite time.

### Lemma L4 (No deadlock under cycle-breaking priority)
Assume A4b: the protocol defines a strict global priority `≺` and ensures that any wait cycle
eventually grants progress to the minimal element under `≺` (others yield/retry).
Then deadlock is impossible even if `WFG(t)` contains cycles.

### Lemma L5 (Global throughput)
Under A1–A4 and finite work per step, the global minimum step
`m(t) = min_z step(z)` increases without bound (the simulation advances).

## 4. Engineering hooks

To make these lemmas operational, the codebase should expose:
- A cycle detector for directed graphs (used in debug/tests).
- Optional debug construction of `WFG(t)` for small runs.
- A stall detector: fail if `m(t)` does not increase for `N` scheduler ticks.

A minimal cycle detector utility is provided in `tdmd/graph_utils.py`.


## A4b IMPLEMENTATION (runtime)
- Implemented as a total-order **priority key** used by the TD automaton when selecting the next computable zone.
- Priority tuple: `(step_id, owner_rank, zone_id)` (owner_rank from holder_map in MPI; 0 in td_local).
- This changes only the *scheduling of enabled transitions* (refinement), not the physical update semantics.


See `docs/WFG_DIAGNOSTICS.md` for local sampled WFG diagnostics.
