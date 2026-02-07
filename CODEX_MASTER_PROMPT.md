# Codex Master Prompt (Current Cycle)

Use this prompt to run Codex on the **current** project cycle.
Historical bootstrap steps for v4.4/v4.8 are complete.

```text
You are working on TDMD-TD (Time Decomposition MD).

Read and obey:
- AGENTS.md
- docs/TODO.md
- docs/ROADMAP.md
- docs/MODE_CONTRACTS.md
- docs/VISUALIZATION.md

Current status:
- Baseline + hardening + v2 risk burndown are complete.
- Universal visualization/analysis track (Phase G, PR-VZ01..PR-VZ08) is implemented.
- Active cycle: GPU portability via Kokkos (Phase H, PR-K01..PR-K10, see `docs/PORTABILITY_KOKKOS_PLAN.md`).
- Work mode: portability implementation in strict PR order without TD semantic drift.

Rules:
- Do not change TD automaton semantics (F/D/P/W/S).
- Do not add implicit global barriers.
- Visualization/output is passive observability only.
- CPU path remains formal reference semantics.

Task:
1) Select ONE highest-priority unchecked item from docs/TODO.md.
2) State selected scope and implement it end-to-end.
3) Add/adjust tests and docs required by AGENTS.md.
4) If selected task is in PR-K cycle, follow `docs/PORTABILITY_KOKKOS_PLAN.md` gate mapping exactly.

Mandatory checks before completion:
- .venv/bin/python -m pytest -q
- .venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict
- .venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --strict
- If ensemble behavior touched:
  - .venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset nvt_smoke --strict
  - .venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset npt_smoke --strict
- If GPU behavior touched:
  - .venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_smoke --strict
  - .venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_interop_smoke --strict
  - .venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_metal_smoke --strict

Completion report format:
1) Scope delivered
2) Changed files
3) Verification command results
4) Contract/risk impact
5) Next logical follow-up
```
