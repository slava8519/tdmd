import json
import os

def test_golden_cfg_system_smoke_file():
    path = os.path.join("golden", "cfg_system_smoke.json")
    assert os.path.exists(path), f"golden file missing: {path}"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("preset") == "smoke"
    rows = data.get("rows", [])
    assert rows, "golden rows missing"
    required = {
        "case", "zones_total", "use_verlet", "verlet_k_steps",
        "chaos_mode", "chaos_delay_prob",
        "max_dr", "max_dv", "max_dE", "max_dT", "max_dP", "ok",
    }
    for r in rows:
        missing = required.difference(r.keys())
        assert not missing, f"golden row missing keys: {missing}"
