from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FloatArr = np.ndarray
IntArr = np.ndarray


@dataclass(frozen=True)
class ZoneView:
    """GPU-friendly, contiguous structure-of-arrays view for a work zone."""

    n: int
    x: FloatArr
    y: FloatArr
    z: FloatArr
    type: IntArr
    id: IntArr


@dataclass(frozen=True)
class HaloView:
    """GPU-friendly, contiguous SoA view for halo atoms (read-only for backends)."""

    n: int
    x: FloatArr
    y: FloatArr
    z: FloatArr
    type: IntArr
    id: IntArr


@dataclass(frozen=True)
class ForceParams:
    cutoff: float
    box: tuple[float, float, float]
    pbc: tuple[bool, bool, bool] = (True, True, True)
    precision: str = "f64"
    potential: dict | None = None


@dataclass(frozen=True)
class ComputeRequest:
    need_energy: bool = False
    need_virial: bool = False
    deterministic: bool = False


@dataclass(frozen=True)
class ComputeResult:
    fx: FloatArr
    fy: FloatArr
    fz: FloatArr
    energy: float | None = None
    virial6: FloatArr | None = None  # (xx,yy,zz,xy,xz,yz)
    backend_diag: dict | None = None


def _as_f64_contig(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return np.ascontiguousarray(a)


def _as_i32_contig(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.int32)
    return np.ascontiguousarray(a)


def build_zone_view(*, x, y, z, type, id) -> ZoneView:
    """Build a contiguous SoA ZoneView from arrays.

    This function should serve as the single 'door' from TD compute to backends.
    """
    x = _as_f64_contig(x)
    y = _as_f64_contig(y)
    z = _as_f64_contig(z)
    t = _as_i32_contig(type)
    i = _as_i32_contig(id)
    n = int(x.shape[0])
    assert y.shape == (n,) and z.shape == (n,)
    assert t.shape == (n,) and i.shape == (n,)
    return ZoneView(n=n, x=x, y=y, z=z, type=t, id=i)


def build_halo_view(*, x, y, z, type, id) -> HaloView:
    x = _as_f64_contig(x)
    y = _as_f64_contig(y)
    z = _as_f64_contig(z)
    t = _as_i32_contig(type)
    i = _as_i32_contig(id)
    n = int(x.shape[0])
    assert y.shape == (n,) and z.shape == (n,)
    assert t.shape == (n,) and i.shape == (n,)
    return HaloView(n=n, x=x, y=y, z=z, type=t, id=i)
