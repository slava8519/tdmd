import numpy as np

from tdmd.geom_pbc import mask_in_aabb_pbc
from tdmd.zones import compute_zone_buffer_skin


def test_invariant_buffer_skin_lag():
    v = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 4.0]], dtype=float)
    ids = np.array([0, 1], dtype=np.int32)
    dt = 0.1
    Kb = 1.2
    lag_steps = 2
    b, skin = compute_zone_buffer_skin(v, ids, dt, Kb, skin_from_buffer=True, lag_steps=lag_steps)
    expected = Kb * 5.0 * dt * (lag_steps + 1)
    assert abs(b - expected) < 1e-12
    assert abs(skin - expected) < 1e-12


def test_invariant_mask_in_aabb_pbc():
    box = 10.0
    r = np.array(
        [
            [3.0, 3.0, 3.0],  # inside
            [9.5, 3.0, 3.0],  # outside
        ],
        dtype=float,
    )
    ids = np.array([0, 1], dtype=np.int32)
    lo = np.array([2.0, 2.0, 2.0], dtype=float)
    hi = np.array([4.0, 4.0, 4.0], dtype=float)
    mask = mask_in_aabb_pbc(r, ids, lo, hi, pad=0.0, box=box)
    assert mask.tolist() == [True, False]
