# GPU Roadmap

Principle: GPU is a **zone compute backend**. TD automaton stays on CPU.

## Phase 0 — Backend interface (CPU-only)
- Add `ZoneComputeBackend` + `CPUBackend`.
- Wire TD compute through backend abstraction.

## Phase 1 — Single-zone GPU (correctness)
- Feature-gated `GPUBackend` (default OFF).
- verify v2 GPU-vs-CPU regression.

## Phase 2 — Overlap (performance)
- Async kernels + comm overlap without breaking `W<=1`.
- Track speedup vs contention metrics (Pareto).


See `docs/GPU_BACKEND_API.md` for the strict backend contract.


Data layout: see `docs/PR0_DATA_CONTRACT.md` (PR-0 golden SoA contract).
