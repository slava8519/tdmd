.PHONY: test verify-smoke gpu-profile eam-decomp-gpu-heavy eam-zone-sweep-gpu eam-td-breakdown-gpu td-autozoning-advisor-gpu al-crack-100k-gpu ml-reference-smoke ml-reference-parity fmt lint typecheck precommit clean-results

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
	$(PY) scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset eam_decomp_perf_smoke --strict

gpu-profile:
	$(PY) scripts/profile_gpu_backend.py --config examples/td_1d_morse.yaml --out-csv results/gpu_profile.csv --out-md results/gpu_profile.md --out-json results/gpu_profile.summary.json --require-effective-cuda --eam-n-atoms 512 --eam-steps 2

eam-decomp-gpu-heavy:
	$(PY) scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset eam_decomp_perf_gpu_heavy --strict

eam-zone-sweep-gpu:
	$(PY) scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset eam_decomp_zone_sweep_gpu --strict

eam-td-breakdown-gpu:
	$(PY) scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset eam_td_breakdown_gpu --strict

td-autozoning-advisor-gpu:
	$(PY) scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset td_autozoning_advisor_gpu --strict

al-crack-100k-gpu:
	$(PY) scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset al_crack_100k_compare_gpu --strict

ml-reference-smoke:
	$(PY) scripts/run_verifylab_matrix.py examples/td_1d_morse.yaml --preset ml_reference_smoke --strict

ml-reference-parity:
	$(PY) scripts/ml_reference_parity_pack.py --fixture examples/interop/ml_reference_suite_v1.json --config examples/td_1d_morse.yaml --strict


fmt:
	$(PY) -m black .


lint:
	$(PY) -m ruff check .


typecheck:
	$(PY) -m mypy --follow-imports=skip tdmd/atoms.py tdmd/output.py tdmd/observer.py tdmd/force_dispatch.py tdmd/cli_parser.py


precommit:
	$(PY) -m pre_commit run --all-files

clean-results:
	rm -rf results/*/
