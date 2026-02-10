from __future__ import annotations

import json

from scripts.run_verifylab_matrix import apply_envelope_gate
from tdmd.verify_lab import LabRow


def _mk_row(*, zones_total: int, final_dr: float, rms_dr: float) -> LabRow:
    return LabRow(
        case="cfg_system",
        zones_total=int(zones_total),
        use_verlet=False,
        verlet_k_steps=10,
        chaos_mode=False,
        chaos_delay_prob=0.0,
        steps=300,
        every=10,
        ok=True,
        max_dr=0.0,
        max_dv=0.0,
        max_dE=0.0,
        max_dT=0.0,
        max_dP=0.0,
        final_dr=float(final_dr),
        final_dv=0.0,
        final_dE=0.0,
        final_dT=0.0,
        final_dP=0.0,
        rms_dr=float(rms_dr),
        rms_dv=0.0,
        rms_dE=0.0,
        rms_dT=0.0,
        rms_dP=0.0,
        details={},
    )


def test_apply_envelope_gate_flags_regression(tmp_path):
    fixture = {
        "envelope_version": 1,
        "rows": [
            {
                "case": "cfg_system",
                "zones_total": 4,
                "use_verlet": False,
                "verlet_k_steps": 10,
                "chaos_mode": False,
                "chaos_delay_prob": 0.0,
                "max": {"final_dr": 1.0, "rms_dr": 1.0},
            }
        ],
    }
    fixture_path = tmp_path / "env.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")

    rows = [_mk_row(zones_total=4, final_dr=2.0, rms_dr=0.5)]
    out = apply_envelope_gate(rows, str(fixture_path))

    assert out["ok_all"] is False
    assert int(out["rows_failed"]) == 1
    assert rows[0].ok is False
    assert any("envelope_final_dr" in v["violation"] for v in out["violations"])
