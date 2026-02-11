from __future__ import annotations

from tdmd.constants import FLOAT_EQ_ATOL, GEOM_EPSILON, NUMERICAL_ZERO


def test_numerical_zero_is_positive_and_tiny():
    assert NUMERICAL_ZERO > 0.0
    assert NUMERICAL_ZERO < 1e-20, "NUMERICAL_ZERO must be far below any physical scale"


def test_float_eq_atol_is_positive_and_tiny():
    assert FLOAT_EQ_ATOL > 0.0
    assert FLOAT_EQ_ATOL < 1e-10


def test_geom_epsilon_is_positive_and_tiny():
    assert GEOM_EPSILON > 0.0
    assert GEOM_EPSILON < 1e-6


def test_ordering_numerical_zero_smallest():
    """NUMERICAL_ZERO < GEOM_EPSILON < FLOAT_EQ_ATOL would be wrong;
    only require all are small and positive."""
    assert NUMERICAL_ZERO < GEOM_EPSILON
    assert GEOM_EPSILON < 1.0


def test_division_guard():
    """NUMERICAL_ZERO must prevent division-by-zero without affecting result."""
    large = 1e10
    result = large / (0.0 + NUMERICAL_ZERO)
    assert result > 0.0
    assert result < float("inf")


def test_float_eq_comparison():
    """FLOAT_EQ_ATOL must distinguish 1.0 from 1.0 + 1e-14 but not 1.0 + 1e-16."""
    a = 1.0
    b = 1.0 + 1e-16
    c = 1.0 + 1e-14
    assert abs(a - b) <= FLOAT_EQ_ATOL, "Values differing by 1e-16 should be equal"
    assert abs(a - c) > FLOAT_EQ_ATOL, "Values differing by 1e-14 should be distinguishable"
