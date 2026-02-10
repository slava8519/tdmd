.PHONY: test verify-smoke fmt lint typecheck precommit

PY ?= .venv/bin/python
ifeq ($(wildcard $(PY)),)
  PY = python3
endif

test:
	$(PY) -m pytest -q


verify-smoke:
	$(PY) -m pytest -q
	$(PY) scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict
	$(PY) scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --strict


fmt:
	$(PY) -m black .


lint:
	$(PY) -m ruff check .


typecheck:
	$(PY) -m mypy --follow-imports=skip tdmd/atoms.py tdmd/output.py tdmd/observer.py tdmd/force_dispatch.py tdmd/cli_parser.py


precommit:
	$(PY) -m pre_commit run --all-files
