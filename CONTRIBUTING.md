# Contributing (Codex-friendly)

## Rules
- One PR = one coherent change.
- If you touch semantics: update `docs/SPEC_TD_AUTOMATON.md` and/or `docs/INVARIANTS.md`.
- Any new invariant must be verified (unit test or verifylab metric).
- Keep changes small, reviewable, and reproducible.

## Quick checks
```bash
python -m pytest -q
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --strict
```
