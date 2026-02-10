from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from tdmd.io.metrics import MetricsWriter
from tdmd.io.trajectory import TrajectoryWriter
from tdmd.viz import MobilityPlugin, RegionOccupancyPlugin, SpeciesMixingPlugin, run_plugins


class _ZeroPotential:
    def pair(self, r2, cutoff2, type_i=None, type_j=None):
        rr = np.asarray(r2, dtype=float)
        return np.zeros_like(rr), np.zeros_like(rr)


def test_trajectory_manifest_and_channels(tmp_path):
    p = tmp_path / "traj.lammpstrj"
    ids = np.array([1, 2], dtype=np.int32)
    types = np.array([1, 2], dtype=np.int32)
    w = TrajectoryWriter(
        str(p),
        box=(10.0, 10.0, 10.0),
        pbc=(True, True, True),
        atom_ids=ids,
        atom_types=types,
        channels=("unwrapped", "image"),
        compression="none",
        write_output_manifest=True,
    )
    r0 = np.array([[9.9, 1.0, 1.0], [0.1, 2.0, 2.0]], dtype=float)
    v0 = np.zeros((2, 3), dtype=float)
    w.write(0, r0, v0)
    r1 = np.array([[0.2, 1.0, 1.0], [9.8, 2.0, 2.0]], dtype=float)
    w.write(1, r1, v0)
    w.close()

    mpath = tmp_path / "traj.lammpstrj.manifest.json"
    assert mpath.exists()
    m = json.loads(mpath.read_text(encoding="utf-8"))
    assert m["schema"]["name"] == "tdmd.trajectory.lammps_dump"
    assert "xu" in m["columns"]
    assert "ix" in m["columns"]

    text = p.read_text(encoding="utf-8").splitlines()
    hdr_idx = [i for i, ln in enumerate(text) if ln.startswith("ITEM: ATOMS ")]
    hdrs = [text[i] for i in hdr_idx]
    assert len(hdrs) == 2
    assert "xu yu zu ix iy iz" in hdrs[0]
    # Second frame atom lines start after second "ITEM: ATOMS ..." header.
    j = hdr_idx[1]
    row1 = text[j + 1].split()
    row2 = text[j + 2].split()
    # id type x y z vx vy vz xu yu zu ix iy iz
    assert row1[0] == "1"
    assert row1[11] == "1"  # ix
    assert abs(float(row1[8]) - 10.2) < 1e-8  # xu
    assert row2[0] == "2"
    assert row2[11] == "-1"  # ix
    assert abs(float(row2[8]) - (-0.2)) < 1e-8  # xu


def test_metrics_manifest(tmp_path):
    p = tmp_path / "metrics.csv"
    r = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=float)
    v = np.zeros((2, 3), dtype=float)
    w = MetricsWriter(
        str(p),
        mass=np.array([1.0, 2.0], dtype=float),
        box=10.0,
        cutoff=2.5,
        potential=_ZeroPotential(),
        atom_types=np.array([1, 1], dtype=np.int32),
        write_output_manifest=True,
    )
    w.write(0, r, v)
    w.close()
    mpath = tmp_path / "metrics.csv.manifest.json"
    assert mpath.exists()
    m = json.loads(mpath.read_text(encoding="utf-8"))
    assert m["schema"]["name"] == "tdmd.metrics.csv"
    assert m["atom_count"] == 2
    assert m["columns"][0] == "step"


def test_viz_plugins_run(tmp_path):
    traj = tmp_path / "tiny.lammpstrj"
    ids = np.array([1, 2, 3], dtype=np.int32)
    types = np.array([1, 2, 2], dtype=np.int32)
    w = TrajectoryWriter(
        str(traj),
        box=(10.0, 10.0, 10.0),
        pbc=(True, True, True),
        atom_ids=ids,
        atom_types=types,
        channels=("unwrapped",),
        write_output_manifest=False,
    )
    v = np.zeros((3, 3), dtype=float)
    w.write(0, np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]), v)
    w.write(1, np.array([[1.2, 1.0, 1.0], [2.1, 2.0, 2.0], [3.0, 3.2, 3.0]]), v)
    w.write(2, np.array([[1.4, 1.0, 1.0], [2.2, 2.0, 2.0], [3.0, 3.4, 3.0]]), v)
    w.close()

    rows, summary = run_plugins(
        str(traj),
        [
            MobilityPlugin(),
            SpeciesMixingPlugin(),
            RegionOccupancyPlugin(xlo=0, xhi=2.5, ylo=0, yhi=2.5, zlo=0, zhi=2.5),
        ],
    )
    assert len(rows) == 3
    assert "mobility.rms" in rows[-1]
    assert "species_mixing.index" in rows[-1]
    assert "region_occupancy.count" in rows[-1]
    assert summary["frames"] == 3


def test_viz_analyze_script(tmp_path):
    traj = tmp_path / "tiny2.lammpstrj"
    ids = np.array([1, 2], dtype=np.int32)
    types = np.array([1, 2], dtype=np.int32)
    w = TrajectoryWriter(
        str(traj),
        box=(10.0, 10.0, 10.0),
        pbc=(True, True, True),
        atom_ids=ids,
        atom_types=types,
        channels=("unwrapped",),
        write_output_manifest=False,
    )
    v = np.zeros((2, 3), dtype=float)
    w.write(0, np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), v)
    w.write(1, np.array([[1.1, 1.0, 1.0], [2.0, 2.2, 2.0]]), v)
    w.close()

    out_csv = tmp_path / "ana.csv"
    out_json = tmp_path / "ana.json"
    cmd = [
        sys.executable,
        "scripts/viz_analyze.py",
        "--traj",
        str(traj),
        "--plugin",
        "mobility",
        "--plugin",
        "species_mixing",
        "--out-csv",
        str(out_csv),
        "--out-json",
        str(out_json),
    ]
    subprocess.check_call(cmd)
    assert out_csv.exists()
    assert out_json.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["rows"] >= 2


def test_trajectory_force_channel(tmp_path):
    p = tmp_path / "traj_force.lammpstrj"
    ids = np.array([1, 2], dtype=np.int32)
    types = np.array([1, 1], dtype=np.int32)
    w = TrajectoryWriter(
        str(p),
        box=(10.0, 10.0, 10.0),
        pbc=(True, True, True),
        atom_ids=ids,
        atom_types=types,
        channels=("force",),
        write_output_manifest=False,
        force_potential=_ZeroPotential(),
        force_cutoff=2.5,
        force_atom_types=types,
    )
    r = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=float)
    v = np.zeros((2, 3), dtype=float)
    w.write(0, r, v)
    w.close()
    lines = p.read_text(encoding="utf-8").splitlines()
    hdr = [ln for ln in lines if ln.startswith("ITEM: ATOMS ")][0]
    assert hdr.endswith("fx fy fz")
    atom_row = lines[lines.index(hdr) + 1].split()
    # id type x y z vx vy vz fx fy fz
    assert len(atom_row) == 11
    assert abs(float(atom_row[8])) < 1e-12


def test_main_run_td_local_outputs_manifest(tmp_path):
    traj = tmp_path / "td_local_traj.lammpstrj"
    metrics = tmp_path / "td_local_metrics.csv"
    cmd = [
        sys.executable,
        "-m",
        "tdmd.main",
        "run",
        "examples/td_1d_morse.yaml",
        "--task",
        "examples/interop/task.yaml",
        "--mode",
        "td_local",
        "--device",
        "cpu",
        "--traj",
        str(traj),
        "--traj-every",
        "1",
        "--traj-channels",
        "unwrapped,image",
        "--metrics",
        str(metrics),
        "--metrics-every",
        "1",
    ]
    subprocess.check_call(cmd)
    assert traj.exists()
    assert (tmp_path / "td_local_traj.lammpstrj.manifest.json").exists()
    assert metrics.exists()
    assert (tmp_path / "td_local_metrics.csv.manifest.json").exists()
