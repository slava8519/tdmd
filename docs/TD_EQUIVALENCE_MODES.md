# TD Equivalence Modes

This document defines equivalence expectations for supported TD execution modes.

## Ring-Transfer (Dissertation Core)
**Name:** `deps_provider_mode=dynamic`  
**Ownership:** zone ownership is transferred along the ring via `REC_ZONE` messages.  
**Messages used:** `REC_ZONE`, `REC_DELTA_MIGRATION`, `REC_DELTA_HALO`, holder/req traffic.  
**Expected equivalence:** matches the dissertation core. Asynchronous TD ordering should remain
numerically close to serial within VerifyLab tolerances (dr/dv/dE/dT/dP), and preserve invariants
I0–I6 (see `docs/INVARIANTS.md`).

## Static Round-Robin (Explicit Extension)
**Name:** `deps_provider_mode=static_rr`  
**Ownership:** fixed `owner_rank(zid) = zid % P`; `REC_ZONE` is used only at startup.  
**Messages used:** `REC_DELTA_MIGRATION`, `REC_DELTA_HALO`, holder/req traffic; no ownership pipeline.  
**Expected equivalence:** asynchronous TD ordering should remain numerically close to serial
within VerifyLab tolerances; invariants I0–I6 still apply. Static ownership can reduce
pipeline churn but is **not** part of the dissertation ring-transfer core.

### Notes
- In `static_rr`, the `SEND_ZONE` transition does **not** transfer ownership; zones remain on
  their owner rank and return to `D` after compute. Only halos/migrations are routed.
- Any new optimization must preserve the formal core behavior, or introduce a new invariant
  plus verification (tests or VerifyLab metrics).
