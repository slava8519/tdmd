# Contracts (Do-not-break rules)

This file defines **hard constraints** for PR cycles, especially when using Codex agents.
If a change violates these constraints, it is considered a semantic regression.

## TD semantics (frozen unless explicitly approved)
- Do not modify the TD automaton semantics (state transitions, phase ordering).
- Do not change deps rules or ownership semantics.
- Do not weaken the formal core invariant: **at most one zone may be in compute state W at a time per rank** (W<=1).
- Do not change liveness policy A4b (total-order tie-break) unless a dedicated liveness PR updates proofs/docs.

## GPU integration constraints
- GPU is a **compute backend refinement** only.
- GPU backend must not:
  - perform MPI,
  - change scheduling,
  - decide zone execution order.

## Verification constraints
- `pytest -q` must remain green.
- verify v2 is the scientific gate. GPU cannot be enabled by default until GPU verify plan phases pass.

## PR-0 constraints (backend abstraction)
- PR-0 is CPU-only. No CUDA kernels, no GPU dependencies required at import time.
- The CPUBackend must match existing CPU force computation semantics.
- Data contract must follow `docs/PR0_DATA_CONTRACT.md` and `tdmd/zone_views.py`.
