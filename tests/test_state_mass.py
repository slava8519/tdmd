from __future__ import annotations

import numpy as np
import pytest

from tdmd.state import kinetic_energy


def test_kinetic_energy_accepts_mass_array():
    v = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    masses = np.array([1.0, 2.0], dtype=float)
    ke = kinetic_energy(v, masses)
    # 0.5 * (1*1^2 + 2*2^2) = 4.5
    assert ke == pytest.approx(4.5)
