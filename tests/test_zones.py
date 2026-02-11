from __future__ import annotations

import warnings

import numpy as np
import pytest

from tdmd.zones import (
    Zone,
    ZoneLayout1DCells,
    ZoneLayout3DBlocks,
    ZoneType,
    assign_atoms_to_zones,
    assign_atoms_to_zones_3d,
    compute_required_buffer,
    compute_zone_buffer_skin,
    zones_overlapping_interval,
    zones_overlapping_range_pbc,
)

# ---------------------------------------------------------------------------
# ZoneType enum
# ---------------------------------------------------------------------------


def test_zone_type_values():
    assert ZoneType.F == "f"
    assert ZoneType.D == "d"
    assert ZoneType.W == "w"
    assert ZoneType.S == "s"
    assert ZoneType.P == "p"


# ---------------------------------------------------------------------------
# Zone.contains
# ---------------------------------------------------------------------------


def test_zone_contains_basic():
    z = Zone(zid=0, z0=2.0, z1=5.0, n_cells=1)
    coords = np.array([1.0, 2.0, 3.0, 4.99, 5.0, 6.0])
    mask = z.contains(coords, box=10.0)
    assert list(mask) == [False, True, True, True, False, False]


def test_zone_contains_pbc():
    """Atom at z=12 in box=10 wraps to z=2, should be inside [2,5)."""
    z = Zone(zid=0, z0=2.0, z1=5.0, n_cells=1)
    coords = np.array([12.0, 15.0])
    mask = z.contains(coords, box=10.0)
    assert list(mask) == [True, False]


# ---------------------------------------------------------------------------
# ZoneLayout1DCells.build
# ---------------------------------------------------------------------------


def test_layout_build_uniform_zones():
    layout = ZoneLayout1DCells(box=10.0, cell_size=1.0, zones_total=2, zone_cells_w=5, zone_cells_s=5)
    zones = layout.build()
    assert len(zones) == 2
    assert zones[0].z0 == 0.0
    assert zones[0].z1 == pytest.approx(5.0)
    assert zones[1].z0 == pytest.approx(5.0)
    assert zones[1].z1 == pytest.approx(10.0)


def test_layout_build_covers_full_box():
    layout = ZoneLayout1DCells(box=10.0, cell_size=1.0, zones_total=3, zone_cells_w=2, zone_cells_s=2)
    zones = layout.build()
    assert zones[0].z0 == 0.0
    assert zones[-1].z1 == pytest.approx(10.0)


def test_layout_build_strict_min_width_raises():
    with pytest.raises(ValueError, match="min_zone_width too large"):
        ZoneLayout1DCells(
            box=6.0,
            cell_size=1.0,
            zones_total=4,
            zone_cells_w=1,
            zone_cells_s=1,
            min_zone_width=3.0,
            strict_min_width=True,
        ).build()


def test_layout_build_min_width_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ZoneLayout1DCells(
            box=6.0,
            cell_size=1.0,
            zones_total=4,
            zone_cells_w=1,
            zone_cells_s=1,
            min_zone_width=3.0,
            strict_min_width=False,
        ).build()
    assert any("min_zone_width" in str(x.message) for x in w)


def test_layout_empty_zones_when_too_many():
    layout = ZoneLayout1DCells(box=3.0, cell_size=1.0, zones_total=5, zone_cells_w=1, zone_cells_s=1)
    zones = layout.build()
    assert len(zones) == 5
    nonempty = [z for z in zones if z.n_cells > 0]
    empty = [z for z in zones if z.n_cells == 0]
    assert len(nonempty) >= 1
    assert len(empty) >= 1


# ---------------------------------------------------------------------------
# assign_atoms_to_zones
# ---------------------------------------------------------------------------


def test_assign_atoms_to_zones():
    layout = ZoneLayout1DCells(box=10.0, cell_size=5.0, zones_total=2, zone_cells_w=1, zone_cells_s=1)
    zones = layout.build()
    # 4 atoms: x,y don't matter; z determines zone
    r = np.array([
        [0.0, 0.0, 1.0],  # zone 0
        [0.0, 0.0, 3.0],  # zone 0
        [0.0, 0.0, 6.0],  # zone 1
        [0.0, 0.0, 9.0],  # zone 1
    ])
    assign_atoms_to_zones(r, zones, box=10.0)
    assert set(zones[0].atom_ids.tolist()) == {0, 1}
    assert set(zones[1].atom_ids.tolist()) == {2, 3}


# ---------------------------------------------------------------------------
# compute_required_buffer / compute_zone_buffer_skin
# ---------------------------------------------------------------------------


def test_compute_required_buffer_formula():
    """buffer = Kb * vmax * dt * (lag + 1)"""
    b = compute_required_buffer(vmax=5.0, dt=0.1, Kb=1.2, lag_steps=2)
    expected = 1.2 * 5.0 * 0.1 * 3
    assert abs(b - expected) < 1e-12


def test_compute_required_buffer_zero_lag():
    b = compute_required_buffer(vmax=5.0, dt=0.1, Kb=1.0, lag_steps=0)
    assert abs(b - 0.5) < 1e-12


def test_compute_zone_buffer_skin_empty():
    v = np.zeros((10, 3))
    ids = np.empty((0,), dtype=np.int32)
    b, skin = compute_zone_buffer_skin(v, ids, dt=0.1, Kb=1.0)
    assert b == 0.0
    assert skin == 0.0


def test_compute_zone_buffer_skin_correct():
    v = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 4.0]], dtype=float)  # vmax = 5.0
    ids = np.array([0, 1], dtype=np.int32)
    b, skin = compute_zone_buffer_skin(v, ids, dt=0.1, Kb=1.2, lag_steps=2)
    expected = 1.2 * 5.0 * 0.1 * 3
    assert abs(b - expected) < 1e-12
    assert abs(skin - expected) < 1e-12


# ---------------------------------------------------------------------------
# zones_overlapping_interval / zones_overlapping_range_pbc
# ---------------------------------------------------------------------------


def test_zones_overlapping_interval_basic():
    zones = [
        Zone(zid=0, z0=0.0, z1=5.0, n_cells=1),
        Zone(zid=1, z0=5.0, z1=10.0, n_cells=1),
    ]
    assert zones_overlapping_interval(4.0, 6.0, zones) == [0, 1]
    assert zones_overlapping_interval(0.0, 3.0, zones) == [0]
    assert zones_overlapping_interval(6.0, 8.0, zones) == [1]


def test_zones_overlapping_range_pbc_no_wrap():
    zones = [
        Zone(zid=0, z0=0.0, z1=5.0, n_cells=1),
        Zone(zid=1, z0=5.0, z1=10.0, n_cells=1),
    ]
    result = zones_overlapping_range_pbc(3.0, 7.0, 10.0, zones)
    assert set(result) == {0, 1}


def test_zones_overlapping_range_pbc_wrapping():
    zones = [
        Zone(zid=0, z0=0.0, z1=5.0, n_cells=1),
        Zone(zid=1, z0=5.0, z1=10.0, n_cells=1),
    ]
    # Query wraps around: z0=-2 (=8 mod 10), z1=2 => hits zone 0 and zone 1
    result = zones_overlapping_range_pbc(-2.0, 2.0, 10.0, zones)
    assert 0 in result
    assert 1 in result


def test_zones_overlapping_range_pbc_zero_width():
    zones = [Zone(zid=0, z0=0.0, z1=10.0, n_cells=1)]
    assert zones_overlapping_range_pbc(5.0, 5.0, 10.0, zones) == []


def test_zones_overlapping_range_pbc_zero_box():
    zones = [Zone(zid=0, z0=0.0, z1=10.0, n_cells=1)]
    assert zones_overlapping_range_pbc(0.0, 5.0, 0.0, zones) == []


# ---------------------------------------------------------------------------
# ZoneLayout3DBlocks
# ---------------------------------------------------------------------------


def test_layout_3d_build():
    layout = ZoneLayout3DBlocks.build(box=10.0, nx=2, ny=2, nz=2)
    assert len(layout.zones) == 8
    # Each zone should have lo < hi in all dimensions
    for z in layout.zones:
        assert all(z.lo[i] < z.hi[i] for i in range(3))


def test_layout_3d_zid_indices_roundtrip():
    layout = ZoneLayout3DBlocks.build(box=10.0, nx=3, ny=2, nz=4)
    for z in layout.zones:
        ix, iy, iz = layout.indices_from_zid(z.zid)
        assert layout.zid_from_indices(ix, iy, iz) == z.zid


def test_layout_3d_zid_from_indices_pbc():
    layout = ZoneLayout3DBlocks.build(box=10.0, nx=3, ny=2, nz=4)
    # Index wrapping
    assert layout.zid_from_indices(3, 0, 0) == layout.zid_from_indices(0, 0, 0)
    assert layout.zid_from_indices(0, 2, 0) == layout.zid_from_indices(0, 0, 0)


def test_assign_atoms_to_zones_3d():
    layout = ZoneLayout3DBlocks.build(box=10.0, nx=2, ny=2, nz=2)
    r = np.array([
        [1.0, 1.0, 1.0],  # zone (0,0,0) = zid 0
        [6.0, 1.0, 1.0],  # zone (1,0,0) = zid 1
        [1.0, 6.0, 1.0],  # zone (0,1,0) = zid 2
        [6.0, 6.0, 6.0],  # zone (1,1,1) = zid 7
    ])
    assign_atoms_to_zones_3d(r, layout)
    assert 0 in layout.zones[0].atom_ids.tolist()
    assert 1 in layout.zones[1].atom_ids.tolist()
    assert 2 in layout.zones[2].atom_ids.tolist()
    assert 3 in layout.zones[7].atom_ids.tolist()
