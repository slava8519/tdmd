from __future__ import annotations

import numpy as np

from tdmd.potentials import make_potential
from tdmd.serial import run_serial
from tdmd.td_local import run_td_local


def _write_linear_table(path: str, keyword: str = "LINEAR") -> None:
    # U(r) = 3 - r, F(r) = -dU/dr = 1
    rows = [
        (1, 1.0, 2.0, 1.0),
        (2, 1.5, 1.5, 1.0),
        (3, 2.0, 1.0, 1.0),
        (4, 2.5, 0.5, 1.0),
        (5, 3.0, 0.0, 1.0),
    ]
    text = ["# linear test table\n", f"{keyword}\n", "N 5 R 1.0 3.0\n"]
    for idx, rr, uu, ff in rows:
        text.append(f"{idx} {rr:.6f} {uu:.6f} {ff:.6f}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(text))


def test_table_potential_interpolation(tmp_path):
    table = tmp_path / "linear.table"
    _write_linear_table(str(table), keyword="LINEAR")
    pot = make_potential("table", {"file": str(table), "keyword": "LINEAR"})
    r = np.array([1.0, 1.25, 2.75, 3.1], dtype=float)
    r2 = r * r
    coef, u = pot.pair(r2, cutoff2=10.0)

    # Inside table range [1.0, 3.0]: linear interpolation for U and constant F=1.
    assert np.isclose(u[0], 2.0)
    assert np.isclose(u[1], 1.75)
    assert np.isclose(u[2], 0.25)
    assert np.isclose(coef[0], 1.0 / 1.0)
    assert np.isclose(coef[1], 1.0 / 1.25)
    assert np.isclose(coef[2], 1.0 / 2.75)
    # Outside table range: force and energy are zero.
    assert np.isclose(u[3], 0.0)
    assert np.isclose(coef[3], 0.0)


def test_table_runtime_serial_vs_td_local_single_zone(tmp_path):
    table = tmp_path / "linear.table"
    _write_linear_table(str(table), keyword="LINEAR")
    pot = make_potential("table", {"file": str(table), "keyword": "LINEAR"})

    box = 10.0
    mass = 1.0
    dt = 0.002
    cutoff = 3.0
    r0 = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.2, 1.0, 1.0],
            [3.4, 1.0, 1.0],
            [4.6, 1.0, 1.0],
        ],
        dtype=float,
    )
    v0 = np.array(
        [
            [0.00, 0.02, 0.00],
            [0.01, 0.00, 0.00],
            [0.00, -0.01, 0.00],
            [-0.01, 0.00, 0.00],
        ],
        dtype=float,
    )

    r_serial = r0.copy()
    v_serial = v0.copy()
    run_serial(r_serial, v_serial, mass, box, pot, dt, cutoff, n_steps=4)

    r_td = r0.copy()
    v_td = v0.copy()
    run_td_local(
        r_td,
        v_td,
        mass,
        box,
        pot,
        dt,
        cutoff,
        n_steps=4,
        cell_size=cutoff,
        zones_total=1,
        zone_cells_w=1,
        zone_cells_s=1,
        traversal="forward",
        use_verlet=False,
        decomposition="1d",
        sync_mode=False,
    )

    assert np.allclose(r_serial, r_td, atol=1e-10, rtol=1e-10)
    assert np.allclose(v_serial, v_td, atol=1e-10, rtol=1e-10)
