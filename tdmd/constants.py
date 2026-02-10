"""Named numeric constants for TDMD-TD.

These constants replace the magic numbers scattered across the codebase.
Any change to these values is a behaviour change and must be verified
via the full VerifyLab strict-gate suite.

Categories
----------
NUMERICAL_ZERO
    Tiny positive guard added before division / sqrt to avoid
    division-by-zero in force kernels.  The value 1e-30 is well below
    any physically meaningful distance (atomic units) and does not
    affect numerics within machine precision.

FLOAT_EQ_ATOL
    Absolute tolerance for floating-point equality comparisons such as
    ``abs(x - y) <= FLOAT_EQ_ATOL``.  Used when comparing zone widths,
    decomposition factors, etc.

GEOM_EPSILON
    Tolerance for geometry / cutoff comparisons.  Used for zone-width
    vs cutoff sanity checks and similar pre-condition guards.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Force-kernel guard (prevent division-by-zero / sqrt-of-zero)
# ---------------------------------------------------------------------------
NUMERICAL_ZERO: float = 1e-30

# ---------------------------------------------------------------------------
# Floating-point near-equality (e.g. zone factor == 1.0)
# ---------------------------------------------------------------------------
FLOAT_EQ_ATOL: float = 1e-15

# ---------------------------------------------------------------------------
# Geometry tolerance (cutoff / zone-width comparisons)
# ---------------------------------------------------------------------------
GEOM_EPSILON: float = 1e-12
