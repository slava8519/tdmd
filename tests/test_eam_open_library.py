from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from tdmd.io import load_task, task_to_arrays, validate_task_for_run
from tdmd.potentials import make_potential
from tdmd.serial import run_serial


def test_open_eam_library_files_load():
    cases = [
        ("examples/potentials/eam_alloy/AlCu.eam.alloy", ["Al", "Cu"]),
        ("examples/potentials/eam_alloy/Al_zhou.eam.alloy", ["Al"]),
        ("examples/potentials/eam_alloy/Cu_mishin1.eam.alloy", ["Cu"]),
        ("examples/potentials/eam_alloy/Cu_zhou.eam.alloy", ["Cu"]),
        ("examples/potentials/eam_alloy/CuNi.eam.alloy", ["Ni", "Cu"]),
        ("examples/potentials/eam_alloy/Fe_Mishin2006.eam.alloy", ["Fe"]),
        ("examples/potentials/eam_alloy/Ni99.eam.alloy", ["Ni"]),
        ("examples/potentials/eam_alloy/Ti_Zhou04.eam.alloy", ["Ti"]),
    ]
    for path, elements in cases:
        pot = make_potential("eam/alloy", {"file": path, "elements": elements})
        assert tuple(pot.elements) == tuple(elements)
        assert int(pot.grid_r.size) > 100
        assert int(pot.grid_rho.size) > 100


def test_open_eam_library_manifest_checksums():
    manifest = Path("examples/potentials/eam_alloy/library.json")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert int(data.get("schema_version", 0)) == 1
    sums_txt = Path("examples/potentials/eam_alloy/SHA256SUMS").read_text(encoding="utf-8")
    sums_map = {}
    for ln in sums_txt.splitlines():
        t = ln.strip()
        if not t:
            continue
        parts = t.split()
        assert len(parts) >= 2
        sums_map[parts[-1]] = parts[0]
    for rec in data.get("potentials", []):
        p = Path(str(rec["file"]))
        assert p.exists()
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h == str(rec["sha256"])
        assert str(rec["file"]) in sums_map
        assert h == sums_map[str(rec["file"])]


def test_crack_generator_writes_eam_block(tmp_path: Path):
    out = tmp_path / "task_small.yaml"
    cmd = [
        sys.executable,
        "scripts/generate_al_cu_crack_task.py",
        "--out",
        str(out),
        "--target-atoms",
        "10000",
        "--box",
        "53.0",
        "--steps",
        "5",
        "--potential-kind",
        "eam/alloy",
        "--eam-file",
        "examples/potentials/eam_alloy/AlCu.eam.alloy",
    ]
    subprocess.check_call(cmd)
    text = out.read_text(encoding="utf-8")
    assert "kind: eam/alloy" in text
    assert "file: examples/potentials/eam_alloy/AlCu.eam.alloy" in text
    assert "elements: [Al, Cu]" in text


def test_new_fe_ni_ti_tasks_run_serial():
    for task_path in (
        "examples/interop/task_eam_fe_mishin2006.yaml",
        "examples/interop/task_eam_ni99.yaml",
        "examples/interop/task_eam_ti_zhou04.yaml",
    ):
        task = load_task(task_path)
        masses = validate_task_for_run(task, allowed_potential_kinds=("eam/alloy",))
        arr = task_to_arrays(task)
        pot = make_potential(task.potential.kind, task.potential.params)
        run_serial(
            arr.r.copy(),
            arr.v.copy(),
            masses,
            float(task.box.x),
            pot,
            float(task.dt),
            float(task.cutoff),
            int(task.steps),
            atom_types=arr.atom_types,
            ensemble_kind=str(task.ensemble.kind),
            thermostat=task.ensemble.thermostat,
            barostat=task.ensemble.barostat,
        )
