# AGENTS.md — Agent Roles for TDMD-TD

Global rules, acceptance criteria, and quality gates are in `CLAUDE.md`.
This file defines agent ownership scopes only.

## Agent: Orchestrator
- Maintains `docs/PROJECT_STATUS.md`, `docs/TODO.md`, `docs/CUDA_EXECUTION_PLAN.md`.
- Splits work into PR-sized tasks in critical order.
- Enforces CI discipline, reproducibility, and strict gates.

## Agent: VerifyLab Engineer
Scope: scientific verification & regression.
- Owns: `tdmd/verify_v2.py`, `tdmd/verify_lab.py`, `scripts/`, `tests/`.
- Delivers strict CI-grade presets and hardware-strict CUDA gates.
- Must not change TD semantics; only verification logic and threshold policy.

## Agent: TD Automaton & Proof
Scope: formal semantics and correctness.
- Owns: `docs/SPEC_TD_AUTOMATON.md`, `docs/INVARIANTS.md`, invariant checks.
- Maps dissertation concepts to code modules.

## Agent: DepsGraph & 3D Scaling
Scope: dependency graphs and geometry.
- Owns: `DepsProvider*`, 3D AABB/PBC logic, general deps-graph (non-ring).
- Must preserve table-deps vs owner-deps separation.

## Agent: IO / Interop Engineer
Scope: task contract and external interoperability.
- Owns: `tdmd/io/*`, CLI I/O flags, interop fixtures under `examples/interop/`.
- Maintains versioned task schema with strict validation.

## Agent: Materials Potentials Engineer
Scope: physical models for metals/alloys.
- Owns: `tdmd/potentials.py`, force kernels, materials testcases.
- Must preserve TD scheduling semantics while extending force-field capabilities.

## Agent: GPU Backend Engineer
Scope: accelerator implementation without semantic drift.
- Owns: backend abstraction, device kernels, GPU execution path.
- Stack: CuPy RawKernel (primary), C++/CUDA extension (Plan B with evidence).
- Must preserve CPU equivalence and avoid new global barriers.

## Agent: Visualization & Analysis Engineer
Scope: universal output contract, post-processing, rendering adapters.
- Owns: `tdmd/io/{trajectory,metrics}.py`, output schema, `scripts/viz_*`.
- Must keep visualization as a passive observability layer.

## Required Documentation Updates (when relevant)
- `docs/SPEC_TD_AUTOMATON.md`, `docs/INVARIANTS.md`, `docs/THEORY_VS_CODE.md`
- `docs/MODE_CONTRACTS.md` (if mode guarantees change)
- `docs/VISUALIZATION.md` (if output schema changes)
- GPU: `docs/GPU_BACKEND.md`, `docs/VERIFYLAB_GPU.md`
