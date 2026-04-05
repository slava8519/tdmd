from __future__ import annotations

from scripts.run_verifylab_matrix import PRESETS as SCRIPT_PRESETS
from tdmd.verifylab.presets import PRESETS, preset_names


def test_verifylab_preset_names_are_sorted_and_complete():
    names = preset_names()
    assert names == tuple(sorted(PRESETS))
    assert "smoke_ci" in names
    assert "gpu_smoke_hw" in names
    assert "eam_decomp_perf_smoke" in names


def test_run_verifylab_matrix_reexports_preset_table():
    assert SCRIPT_PRESETS is PRESETS
