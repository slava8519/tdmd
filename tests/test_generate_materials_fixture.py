from __future__ import annotations

import json
import subprocess
import sys


def test_generate_materials_parity_fixture_script(tmp_path):
    out = tmp_path / "materials_parity_v2_gen.json"
    cmd = [
        sys.executable,
        "scripts/generate_materials_parity_fixture.py",
        "--template",
        "examples/interop/materials_parity_suite_v2_template.json",
        "--config",
        "examples/td_1d_morse.yaml",
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    data = json.loads(out.read_text(encoding="utf-8"))
    assert int(data["suite_version"]) == 2
    assert len(data.get("cases", [])) >= 6
    assert all("expected" in c for c in data.get("cases", []))
