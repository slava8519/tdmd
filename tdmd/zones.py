from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class ZoneType(str, Enum):
    F = "f"
    D = "d"
    P = "p"
    W = "w"
    S = "s"


@dataclass
class Zone:
    zid: int
    z0: float
    z1: float
    n_cells: int
    ztype: ZoneType = ZoneType.F
    atom_ids: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int32))
    buffer: float = 0.0
    skin: float = 0.0

    def contains(self, z: np.ndarray, box: float) -> np.ndarray:
        zz = z % box
        return (zz >= self.z0) & (zz < self.z1)


@dataclass
class ZoneLayout1DCells:
    box: float
    cell_size: float
    zones_total: int
    pattern_cells: Optional[List[int]] = None
    zone_cells_w: int = 1
    zone_cells_s: int = 1
    min_zone_width: float = 0.0
    strict_min_width: bool = False

    def build(self) -> List[Zone]:
        n_cells_total = max(1, int(round(self.box / self.cell_size)))
        dz_cell = self.box / n_cells_total

        if self.pattern_cells is None:
            pat = [max(1, self.zone_cells_w), max(1, self.zone_cells_s)]
        else:
            pat = [max(1, int(x)) for x in self.pattern_cells] or [1]

        min_width = float(self.min_zone_width)
        if min_width > 0.0:
            min_cells = max(1, int(np.ceil(min_width / dz_cell)))
            if min_cells * int(self.zones_total) > n_cells_total:
                msg = (
                    f"min_zone_width too large for zones_total; some zones will be empty "
                    f"(min_cells={min_cells}, total_cells={n_cells_total}, zones_total={self.zones_total})"
                )
                if self.strict_min_width:
                    raise ValueError(msg)
                warnings.warn(msg, RuntimeWarning)
            pat = [max(p, min_cells) for p in pat]

        zones = []
        cell_cursor = 0
        for zid in range(self.zones_total):
            if cell_cursor >= n_cells_total:
                zones.append(Zone(zid=zid, z0=self.box, z1=self.box, n_cells=0, ztype=ZoneType.F))
                continue
            ncz = pat[zid % len(pat)]
            ncz = min(ncz, n_cells_total - cell_cursor)
            z0 = cell_cursor * dz_cell
            z1 = (cell_cursor + ncz) * dz_cell
            zones.append(Zone(zid=zid, z0=z0, z1=z1, n_cells=ncz, ztype=ZoneType.F))
            cell_cursor += ncz

        if cell_cursor < n_cells_total and zones:
            zones[-1].z1 = self.box
            zones[-1].n_cells += n_cells_total - cell_cursor

        if min_width > 0.0:
            bad = []
            for z in zones:
                if z.n_cells <= 0 or z.z1 <= z.z0:
                    continue
                width = float(z.z1 - z.z0)
                if width + 1e-12 < min_width:
                    bad.append((int(z.zid), width))
            if bad:
                zlist = ", ".join([f"{zid}:{w:.6g}" for zid, w in bad[:4]])
                msg = f"Zone width < min_zone_width ({min_width:.6g}): {zlist}"
                if self.strict_min_width:
                    raise ValueError(msg)
                warnings.warn(msg, RuntimeWarning)

        return zones


def assign_atoms_to_zones(r: np.ndarray, zones: List[Zone], box: float):
    z = r[:, 2]
    for zn in zones:
        if zn.n_cells <= 0 or zn.z0 >= zn.z1:
            zn.atom_ids = np.empty((0,), dtype=np.int32)
        else:
            zn.atom_ids = np.where(zn.contains(z, box))[0].astype(np.int32)


def compute_required_buffer(vmax: float, dt: float, Kb: float, lag_steps: int = 0) -> float:
    """Минимально-достаточный буфер для TD по времени.

    Идея: атомы могут «ждать» L шагов до обработки соседней зоны.
    Чтобы таблицы взаимодействия/окрестности оставались корректными, буфер должен
    покрывать максимальное смещение за это время. Для прототипа берем линейную оценку:
        buffer = Kb * vmax * dt * (lag_steps + 1)

    lag_steps=0 даёт классический случай (без временного лага).
    """
    L = max(0, int(lag_steps))
    return float(Kb) * float(vmax) * float(dt) * float(L + 1)


def compute_zone_buffer_skin(
    v: np.ndarray,
    ids: np.ndarray,
    dt: float,
    Kb: float,
    skin_from_buffer: bool = True,
    lag_steps: int = 0,
) -> Tuple[float, float]:
    if ids.size == 0:
        return 0.0, 0.0
    speeds = np.linalg.norm(v[ids], axis=1)
    vmax = float(speeds.max()) if speeds.size else 0.0
    b = compute_required_buffer(vmax=vmax, dt=dt, Kb=Kb, lag_steps=lag_steps)
    skin = b if skin_from_buffer else b
    return b, skin


def zones_overlapping_interval(z0: float, z1: float, zones: List[Zone]) -> List[int]:
    ids = []
    for z in zones:
        if z.n_cells <= 0 or z.z0 >= z.z1:
            continue
        if (z.z1 > z0) and (z.z0 < z1):
            ids.append(z.zid)
    return ids


def zones_overlapping_range_pbc(z0: float, z1: float, box: float, zones: List[Zone]) -> List[int]:
    if box <= 0:
        return []
    width = z1 - z0
    if width <= 0:
        return []
    z0m = z0 % box
    z1m = z0m + width
    if z1m <= box:
        return zones_overlapping_interval(z0m, z1m, zones)
    ids1 = zones_overlapping_interval(z0m, box, zones)
    ids2 = zones_overlapping_interval(0.0, z1m - box, zones)
    return sorted(set(ids1).union(ids2))


# -----------------------------
# 3D block zone layout (v3.9)
# -----------------------------


@dataclass
class Zone3D:
    """3D axis-aligned zone block with periodic box [0,box)^3.

    We keep ZoneType + atom_ids to reuse TD-automaton semantics.
    """

    zid: int
    lo: np.ndarray  # (3,)
    hi: np.ndarray  # (3,)
    idx: Tuple[int, int, int]  # (ix,iy,iz)
    n_cells: int
    ztype: ZoneType = ZoneType.F
    atom_ids: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int32))
    buffer: float = 0.0
    skin: float = 0.0

    def contains(self, r: np.ndarray, box: float) -> np.ndarray:
        rr = r % box
        return (
            (rr[:, 0] >= self.lo[0])
            & (rr[:, 0] < self.hi[0])
            & (rr[:, 1] >= self.lo[1])
            & (rr[:, 1] < self.hi[1])
            & (rr[:, 2] >= self.lo[2])
            & (rr[:, 2] < self.hi[2])
        )


@dataclass
class ZoneLayout3DBlocks:
    box: float
    nx: int
    ny: int
    nz: int
    zones: List[Zone3D]

    @staticmethod
    def build(box: float, nx: int, ny: int, nz: int) -> "ZoneLayout3DBlocks":
        assert nx > 0 and ny > 0 and nz > 0
        dx, dy, dz = box / nx, box / ny, box / nz
        zones: List[Zone3D] = []
        zid = 0
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    lo = np.array([ix * dx, iy * dy, iz * dz], dtype=np.float64)
                    hi = np.array([(ix + 1) * dx, (iy + 1) * dy, (iz + 1) * dz], dtype=np.float64)
                    zones.append(Zone3D(zid=zid, lo=lo, hi=hi, idx=(ix, iy, iz), n_cells=1))
                    zid += 1
        return ZoneLayout3DBlocks(box=box, nx=nx, ny=ny, nz=nz, zones=zones)

    def zid_from_indices(self, ix: int, iy: int, iz: int) -> int:
        ix %= self.nx
        iy %= self.ny
        iz %= self.nz
        return iz * (self.nx * self.ny) + iy * self.nx + ix

    def indices_from_zid(self, zid: int) -> Tuple[int, int, int]:
        nxy = self.nx * self.ny
        iz = zid // nxy
        rem = zid - iz * nxy
        iy = rem // self.nx
        ix = rem - iy * self.nx
        return int(ix), int(iy), int(iz)


def assign_atoms_to_zones_3d(r: np.ndarray, layout: ZoneLayout3DBlocks) -> None:
    """Assign atoms to 3D zones (in-place on layout.zones)."""
    # clear
    for z in layout.zones:
        z.atom_ids = np.empty((0,), dtype=np.int32)
        z.ztype = ZoneType.F
    rr = r % layout.box
    dx, dy, dz = layout.box / layout.nx, layout.box / layout.ny, layout.box / layout.nz
    ix = np.floor(rr[:, 0] / dx).astype(np.int32)
    iy = np.floor(rr[:, 1] / dy).astype(np.int32)
    iz = np.floor(rr[:, 2] / dz).astype(np.int32)
    # clamp edge case rr==box
    ix = np.clip(ix, 0, layout.nx - 1)
    iy = np.clip(iy, 0, layout.ny - 1)
    iz = np.clip(iz, 0, layout.nz - 1)
    zid = iz * (layout.nx * layout.ny) + iy * layout.nx + ix
    # group
    order = np.argsort(zid, kind="mergesort")
    zid_sorted = zid[order]
    ids_sorted = order.astype(np.int32)
    # split by unique
    uniq, start_idx = np.unique(zid_sorted, return_index=True)
    for k, uz in enumerate(uniq):
        s = start_idx[k]
        e = start_idx[k + 1] if k + 1 < len(start_idx) else len(zid_sorted)
        atom_ids = ids_sorted[s:e]
        z = layout.zones[int(uz)]
        z.atom_ids = atom_ids
        z.ztype = ZoneType.D if atom_ids.size else ZoneType.F


def zones_overlapping_aabb_pbc(
    lo: np.ndarray, hi: np.ndarray, box: float, layout: ZoneLayout3DBlocks
) -> List[int]:
    """Return zone ids whose blocks intersect the periodic AABB [lo,hi] (lo/hi in unwrapped coords).

    We treat `lo` and `hi` as possibly outside [0,box) due to expansion by cutoff.
    The query AABB may wrap; we split each dimension into 1 or 2 intervals in [0,box).
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    assert lo.shape == (3,) and hi.shape == (3,)

    # intervals in each dimension after mod
    def intervals_1d(a0, a1):
        # normalize to [0,box) but keep width
        w = a1 - a0
        # If query covers whole box, return whole.
        if w >= box - 1e-12:
            return [(0.0, box)]
        b0 = a0 % box
        b1 = b0 + w
        if b1 <= box:
            return [(b0, b1)]
        else:
            return [(b0, box), (0.0, b1 - box)]

    ix_int = intervals_1d(lo[0], hi[0])
    iy_int = intervals_1d(lo[1], hi[1])
    iz_int = intervals_1d(lo[2], hi[2])

    dx, dy, dz = box / layout.nx, box / layout.ny, box / layout.nz
    deps = set()

    def idx_range(intervals, d, n):
        out = set()
        for a0, a1 in intervals:
            i0 = int(np.floor(a0 / d))
            i1 = int(np.ceil(a1 / d)) - 1
            i0 = max(0, min(n - 1, i0))
            i1 = max(0, min(n - 1, i1))
            for i in range(i0, i1 + 1):
                out.add(i)
        return sorted(out)

    xs = idx_range(ix_int, dx, layout.nx)
    ys = idx_range(iy_int, dy, layout.ny)
    zs = idx_range(iz_int, dz, layout.nz)
    for iz in zs:
        for iy in ys:
            for ix in xs:
                deps.add(layout.zid_from_indices(ix, iy, iz))
    return sorted(deps)
