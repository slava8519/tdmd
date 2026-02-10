# Contributing (Codex-friendly)

## Rules
- One PR = one coherent change.
- If you touch semantics: update `docs/SPEC_TD_AUTOMATON.md` and/or `docs/INVARIANTS.md`.
- Any new invariant must be verified (unit test or verifylab metric).
- Keep changes small, reviewable, and reproducible.

## Quick checks
```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict
.venv/bin/python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --strict
```

## Dev tooling (recommended)
```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pre_commit install
.venv/bin/python -m pre_commit run --all-files
```

## Versioning
- Releases are Git tags + GitHub releases.
- The changelog lives in `RELEASE_NOTES.md` (keep `README.md` current-state).
