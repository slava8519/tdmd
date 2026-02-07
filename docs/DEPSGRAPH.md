# Deps Graph

This document defines required assumptions for general (non-ring) dependency graphs and
what the code guarantees vs. what is only verified.

## Assumptions
- **PBC geometry:** deps are computed under periodic boundary conditions.
- **No global barriers:** readiness is decided per-zone (table/owner deps + time-lag).
- **Owner routing is direct:** messages route to `owner_rank(zid)` (or `holder_map[zid]`), not to prev/next.
- **deps_table / deps_owner exclude self:** caller must include `A(z)` (local atoms) explicitly in table/support.

## Guarantees (by construction)
- `deps_table(z)` returns zones whose AABB overlaps `p`-neighborhood (cutoff) of `z`.
- `deps_owner(z)` returns zones that may receive migrations (buffer-based neighborhood).
- `owner_rank(z)` is stable for static providers (e.g., round-robin mapping).

## Verified Properties (tests / counters)
- **No self-deps**: unit tests for `DepsProvider3DBlock` ensure `z` is excluded from deps lists.
- **Sanity on small grids**: deps ids are in-range and unique on small grids.
- **Owner stability**: `owner_rank(z)` is consistent and within `[0, mpi_size)`.
- **3D invariants**: `tG3/hG3/hV3` counters detect geometric/support violations in halo and table support.

## Non-Ring Note
For `static_3d`, deps are geometry-based and routing uses owner/holder maps.
No part of the dependency definition assumes ring adjacency or prev/next ranks.

## Static Round-Robin (static_rr)
Static ownership mode: each zone is owned by `owner_rank(zid)=zid % P`.
No REC_ZONE pipeline transfers are performed; only HALO and MIGRATION deltas
are routed to the owner/holder.
