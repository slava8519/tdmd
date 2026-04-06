# Contributing

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pre-commit install
pre-commit run --all-files
```

## Testing

```bash
make test           # pytest -q
make verify-smoke   # pytest + smoke_ci + interop_smoke + eam_decomp_perf strict
```

Run the full VerifyLab matrix for specific scopes:
```bash
# Materials
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset metal_smoke --strict
# GPU
python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset gpu_smoke --strict
```

## Linting and Formatting

```bash
make fmt        # black
make lint       # ruff check
make typecheck  # mypy typed island
```

Rules:
- **black** (line-length 100, skip-string-normalization)
- **ruff** (rules: I, F, UP, B)
- **mypy** typed island: `atoms.py`, `output.py`, `observer.py`, `force_dispatch.py`, `cli_parser.py`

## Pull Request Guidelines

- One PR = one coherent change.
- Mandatory gates before merge: `make test` and `make verify-smoke`.
- If you touch semantics: update `docs/SPEC_TD_AUTOMATON.md` and/or `docs/INVARIANTS.md`.
- Any new invariant must be verified (unit test or VerifyLab metric/counter).
- Keep changes small, reviewable, and reproducible.
- GPU changes are refinement only — CPU path remains the formal reference.

## AI Agents

See `CLAUDE.md` for AI agent instructions and `AGENTS.md` for role definitions.

## Versioning

- Releases are Git tags + GitHub releases.
- The changelog lives in `RELEASE_NOTES.md` (keep `README.md` current-state).
