from __future__ import annotations

import subprocess
import sys

import numpy as np
import yaml

from tdmd.config import load_config
from tdmd.td_automaton import TDAutomaton1W, ZoneRuntime
from tdmd.zone_bins_localz import PersistentZoneLocalZBinsCache
from tdmd.zones import ZoneType


def test_config_time_block_k_alias(tmp_path):
    cfg = {
        "system": {"n_atoms": 8, "mass": 1.0, "box": 10.0, "temperature": 1.0, "seed": 1},
        "potential": {"kind": "lj", "params": {"epsilon": 1.0, "sigma": 1.0}},
        "run": {"dt": 0.005, "n_steps": 10, "thermo_every": 5, "cutoff": 2.5},
        "td": {
            "cell_size": 1.0,
            "zones_total": 4,
            "zone_cells_w": 1,
            "zone_cells_s": 1,
            "traversal": "snake",
            "time_block_k": 3,
        },
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    parsed = load_config(str(path))
    assert parsed.td.batch_size == 3


def test_pop_send_batch_tracks_batch_metrics():
    zones = [
        ZoneRuntime(zid=i, z0=float(i), z1=float(i + 1), ztype=ZoneType.D, atom_ids=np.empty((0,), np.int32))
        for i in range(6)
    ]
    autom = TDAutomaton1W(
        zones_runtime=zones,
        box=6.0,
        cutoff=1.0,
        bins_cache=PersistentZoneLocalZBinsCache(),
        traversal_order=list(range(6)),
        formal_core=True,
    )
    autom.send_queue = [0, 1, 2, 3, 4]

    b1 = autom.pop_send_batch(2)
    b2 = autom.pop_send_batch(2)
    b3 = autom.pop_send_batch(2)

    assert b1 == [0, 1]
    assert b2 == [2, 3]
    assert b3 == [4]
    assert autom.diag["send_batches"] == 3
    assert autom.diag["send_batch_zones_total"] == 5
    assert autom.diag["send_batch_size_max"] == 2


def test_bench_time_blocking_dry_run():
    cmd = [
        sys.executable,
        "scripts/bench_time_blocking.py",
        "--config",
        "examples/td_1d_morse_static_rr.yaml",
        "--k-list",
        "1,2,4",
        "--dry-run",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "k_list=[1, 2, 4]" in proc.stdout
