# Codex runbook

## How to run
1. Open repo root in Codex.
2. For fresh machine bootstrap, use:
   - `CODEX_ENV_BOOTSTRAP_PROMPT.md`
   - or run `bash scripts/bootstrap_codex_env.sh`
3. Use one of:
   - `CODEX_MASTER_PROMPT.md`
   - `CODEX_GPU_MASTER_PROMPT.md`
4. Let Codex produce Orchestrator plan + agent tasks, then implement in the same run.
5. For GPU implementation scope, use CUDA-cycle planning docs:
   - `docs/CUDA_EXECUTION_PLAN.md`
   - `docs/PR_PLAN_GPU.md`

## Tooling notes
- Ubuntu package names are `fdfind` and `batcat`; bootstrap config adds aliases:
  - `fd -> fdfind`
  - `bat -> batcat`

## Guardrails
- Never change TD semantics unless explicitly required by docs.
- Keep `pytest -q` green.
