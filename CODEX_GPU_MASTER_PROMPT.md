# Codex Master Prompt — GPU Cycle (CUDA-Only)

```text
You are working on TDMD-TD (Time Decomposition Molecular Dynamics).

CRITICAL:
- Read and obey AGENTS.md.
- Do NOT change TD automaton semantics or scheduling rules.
- GPU work is compute-backend refinement only.
- Current GPU strategy is CUDA-only.

Primary docs:
- docs/PROJECT_STATUS.md
- docs/ROADMAP_GPU.md
- docs/PR_PLAN_GPU.md
- docs/CUDA_EXECUTION_PLAN.md
- docs/GPU_BACKEND_API.md
- docs/MODE_CONTRACTS.md
- docs/CONTRACTS.md
- docs/INVARIANTS.md
- docs/LIVENESS.md
- docs/WFG_DIAGNOSTICS.md
- docs/TODO.md

GOAL:
Implement the next open PR in the CUDA cycle (PR-C*), starting from PR-C02 unless docs alignment is pending.

Stack policy:
- Primary: numba-cuda.
- Plan B (after strict stabilization): CuPy RawKernel or C++/CUDA extension.

Agents:
- Orchestrator (plan + assign)
- Agent A: CUDA backend implementation (numba-cuda first)
- Agent B: tests + VerifyLab strict regression
- Agent C: docs/config/devex and backend evidence
```
