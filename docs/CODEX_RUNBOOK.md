# Codex runbook

## How to run
1. Open repo root in Codex.
2. Use one of:
   - `CODEX_MASTER_PROMPT.md`
   - `CODEX_GPU_MASTER_PROMPT.md`
3. Let Codex produce Orchestrator plan + agent tasks, then implement in the same run.

## Guardrails
- Never change TD semantics unless explicitly required by docs.
- Keep `pytest -q` green.
