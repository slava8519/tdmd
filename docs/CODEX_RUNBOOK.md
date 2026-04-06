# Codex runbook

This is the canonical human-facing entrypoint for running Codex on TDMD.

## Start
1. Fresh machine only:
   - `bash scripts/bootstrap_codex_env.sh`
2. Recommended restart command:
   - `bash scripts/run_codex_tdmd.sh`

`run_codex_tdmd.sh` starts Codex in the repository root with:
- `danger-full-access`
- `approval never`

That avoids the recurring sandbox/escalation pauses during normal local work on this machine.

## Prompts
- Default prompt: `CODEX_MASTER_PROMPT.md`
- GPU-focused prompt: `CODEX_GPU_MASTER_PROMPT.md`

## Read Order
1. `AGENTS.md`
2. `docs/PROJECT_STATUS.md`
3. `docs/TODO.md`
4. `docs/CUDA_EXECUTION_PLAN.md`
5. `docs/MODE_CONTRACTS.md`
6. `docs/VERIFYLAB.md`

## Active Planning Docs
- `docs/PROJECT_STATUS.md`: current state and next active track
- `docs/TODO.md`: PR-sized queue
- `docs/CUDA_EXECUTION_PLAN.md`: detailed GPU / post-CUDA execution plan

## Historical Docs
- `docs/RISK_BURNDOWN_PLAN.md`
- `docs/RISK_BURNDOWN_PLAN_V2.md`

These are retained for provenance, not as the primary active roadmap.

## Tooling Notes
- Ubuntu package names are `fdfind` and `batcat`; bootstrap adds:
  - `fd -> fdfind`
  - `bat -> batcat`
  - `codex-tdmd -> bash <repo>/scripts/run_codex_tdmd.sh`

## Guardrails
- Never change TD semantics unless the active contracts explicitly require it.
- Keep strict gates green; non-strict diagnostic runs do not count.
