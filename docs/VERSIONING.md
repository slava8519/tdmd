# Versioning

TDMD-TD uses **single-source versioning**.

- Canonical version: repository root `VERSION`.
- `tdmd.__version__` reads from `VERSION`.

## Release bump checklist
1. Edit `VERSION`.
2. Prepend section to `RELEASE_NOTES.md`.
3. Run `pytest -q`.
