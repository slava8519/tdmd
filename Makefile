.PHONY: test
test:
	python -m pytest -q


verify-smoke:
	python -m pytest -q
	python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset smoke_ci --strict
	python scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset interop_smoke --strict
