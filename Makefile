.PHONY: test verify-smoke

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
