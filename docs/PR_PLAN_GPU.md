# GPU PR Plan (for Codex)

Each PR must keep `pytest -q` green. GPU work MUST NOT change TD semantics.

## PR-0: Backend abstraction (no GPU yet)
- Add `tdmd/backends/base.py` (ZoneComputeBackend).
- Add `tdmd/backends/cpu.py` (CPUBackend).
- Wire td_local + td_full_mpi compute path via backend.
- Tests: backend equivalence + verify v2 smoke.

## PR-1: GPU scaffold (feature-gated)
- Add `tdmd/backends/gpu.py` placeholder (import-safe).
- Config flag `backend: cpu|gpu` (default cpu).

## PR-2: Single-zone GPU correctness
- Implement GPU forces/energy for one work zone.
- verify v2 GPU-vs-CPU regression case.

## PR-3: Overlap + verifylab integration
- Async kernel + host state machine.
- Export timing + contention metrics; compare Pareto.


See `docs/GPU_BACKEND_API.md` for the strict backend contract.


Data layout: see `docs/PR0_DATA_CONTRACT.md` (PR-0 golden SoA contract).
