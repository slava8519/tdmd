from __future__ import annotations
import numpy as np

def within_interval_pbc(x: np.ndarray, lo: float, hi: float, box: float) -> np.ndarray:
    lo = float(lo); hi = float(hi); box = float(box)
    if lo <= hi:
        return (x >= lo) & (x < hi)
    return (x >= lo) | (x < hi)

def mask_in_aabb_pbc(r: np.ndarray, ids: np.ndarray, lo: np.ndarray, hi: np.ndarray, pad: float, box: float) -> np.ndarray:
    pad = float(pad); box = float(box)
    lo = np.asarray(lo, dtype=float); hi = np.asarray(hi, dtype=float)
    lo2 = (lo - pad) % box
    hi2 = (hi + pad) % box
    rr = r[ids]
    m0 = within_interval_pbc(rr[:,0], lo2[0], hi2[0], box)
    m1 = within_interval_pbc(rr[:,1], lo2[1], hi2[1], box)
    m2 = within_interval_pbc(rr[:,2], lo2[2], hi2[2], box)
    return m0 & m1 & m2
