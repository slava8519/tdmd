from __future__ import annotations
import numpy as np
import subprocess
import pytest
import yaml

from tdmd.io import load_task, export_lammps_data, export_lammps_in, import_lammps_data

def test_lammps_roundtrip(tmp_path):
    task = load_task("examples/interop/task.yaml")
    data_path = tmp_path / "data.lammps"
    export_lammps_data(task, str(data_path))
    data = import_lammps_data(str(data_path))

    assert len(data.atoms) == len(task.atoms)
    assert abs((data.xhi - data.xlo) - task.box.x) < 1e-8
    assert abs((data.yhi - data.ylo) - task.box.y) < 1e-8
    assert abs((data.zhi - data.zlo) - task.box.z) < 1e-8

    task_masses = {}
    for a in task.atoms:
        task_masses.setdefault(int(a.type), float(a.mass))
    assert data.masses == task_masses

    task_atoms = {int(a.id): a for a in task.atoms}
    for a in data.atoms:
        t = task_atoms[int(a.id)]
        assert int(a.type) == int(t.type)
        assert np.allclose(np.array(a.r), np.array(t.r), atol=1e-8)
        assert np.allclose(np.array(a.v), np.array(t.v), atol=1e-8)


def test_lammps_in_export_includes_pair_matrix_coeffs(tmp_path):
    task = {
        "task_version": 1,
        "units": "lj",
        "box": {"x": 10.0, "y": 10.0, "z": 10.0, "pbc": [True, True, True]},
        "atoms": [
            {"id": 1, "type": 1, "mass": 1.0, "r": [1.0, 1.0, 1.0], "v": [0.0, 0.0, 0.0]},
            {"id": 2, "type": 2, "mass": 2.0, "r": [2.0, 2.0, 2.0], "v": [0.0, 0.0, 0.0]},
        ],
        "potential": {
            "kind": "lj",
            "params": {
                "pair_coeffs": {
                    "1-1": {"epsilon": 1.0, "sigma": 1.0},
                    "1-2": {"epsilon": 0.5, "sigma": 1.1},
                    "2-2": {"epsilon": 0.8, "sigma": 0.9},
                }
            },
        },
        "cutoff": 2.5,
        "dt": 0.005,
        "steps": 10,
    }
    task_path = tmp_path / "task_pair_matrix.yaml"
    task_path.write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")
    parsed = load_task(str(task_path))
    in_path = tmp_path / "in.lammps"
    export_lammps_in(parsed, str(in_path), data_filename="data.lammps")
    text = in_path.read_text(encoding="utf-8")
    assert "pair_style lj/cut" in text
    assert "pair_coeff 1 1 1.00000000 1.00000000" in text
    assert "pair_coeff 1 2 0.50000000 1.10000000" in text
    assert "pair_coeff 2 2 0.80000000 0.90000000" in text


def test_lammps_in_export_eam_alloy_coeff_line(tmp_path):
    task = load_task("examples/interop/task_eam_alloy.yaml")
    in_path = tmp_path / "in.lammps"
    export_lammps_in(task, str(in_path), data_filename="data.lammps")
    text = in_path.read_text(encoding="utf-8")
    assert "pair_style eam/alloy" in text
    assert "pair_coeff * *" in text
    assert "eam_al_ni_demo.setfl Al Ni" in text


def test_lammps_data_export_rejects_non_contiguous_types(tmp_path):
    task = {
        "task_version": 1,
        "units": "lj",
        "box": {"x": 10.0, "y": 10.0, "z": 10.0, "pbc": [True, True, True]},
        "atoms": [
            {"id": 1, "type": 1, "mass": 1.0, "r": [1.0, 1.0, 1.0], "v": [0.0, 0.0, 0.0]},
            {"id": 2, "type": 3, "mass": 1.0, "r": [2.0, 2.0, 2.0], "v": [0.0, 0.0, 0.0]},
        ],
        "potential": {"kind": "lj", "params": {"epsilon": 1.0, "sigma": 1.0}},
        "cutoff": 2.5,
        "dt": 0.005,
        "steps": 10,
    }
    task_path = tmp_path / "task_noncontig.yaml"
    task_path.write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")
    parsed = load_task(str(task_path))
    with pytest.raises(ValueError, match="contiguous atom types"):
        export_lammps_data(parsed, str(tmp_path / "data.lammps"))


def test_cli_export_lammps_eam_alloy(tmp_path):
    outdir = tmp_path / "out"
    cmd = [
        ".venv/bin/python",
        "-m",
        "tdmd.main",
        "run",
        "--task",
        "examples/interop/task_eam_alloy.yaml",
        "--mode",
        "serial",
        "--export-lammps",
        str(outdir),
        "--export-only",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    data_path = outdir / "data.lammps"
    in_path = outdir / "in.lammps"
    assert data_path.is_file()
    assert in_path.is_file()
    in_text = in_path.read_text(encoding="utf-8")
    assert "pair_style eam/alloy" in in_text
