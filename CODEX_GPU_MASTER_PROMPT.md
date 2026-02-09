# Codex Master Prompt — GPU Cycle (v4.5.0)

```text
You are working on TDMD-TD (Time Decomposition Molecular Dynamics).

CRITICAL:
- Read and obey AGENTS.md.
- Do NOT change TD automaton semantics or scheduling rules.
- GPU work is compute-backend refinement only.

Primary docs:
- docs/PROJECT_STATUS.md
- docs/ROADMAP_GPU.md
- docs/PR_PLAN_GPU.md
- docs/GPU_BACKEND_API.md
- docs/MODE_CONTRACTS.md
- docs/CONTRACTS.md
- docs/INVARIANTS.md
- docs/LIVENESS.md
- docs/WFG_DIAGNOSTICS.md
- docs/TODO.md

GOAL:
Implement PR-0 (backend abstraction) from docs/PR_PLAN_GPU.md.

Agents:
- Orchestrator (plan + assign)
- Agent A: backend interface + CPUBackend + wiring
- Agent B: tests + verify v2 regression
- Agent C: docs/config/devex (default cpu)
```


NOTE TO ORCHESTRATOR: Agent A should implement/confirm `tdmd/zone_views.py` as the single door into backend compute.
