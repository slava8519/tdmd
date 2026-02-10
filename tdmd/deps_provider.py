from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


@dataclass(frozen=True)
class ZoneGeom1D:
    z0: float
    z1: float


@dataclass(frozen=True)
class ZoneGeomAABB:
    lo: np.ndarray  # (3,)
    hi: np.ndarray  # (3,)
    z0: float = 0.0
    z1: float = 0.0


class DepsProvider(Protocol):
    def deps_table(self, zid: int) -> list[int]: ...
    def deps_owner(self, zid: int) -> list[int]: ...
    def owner_rank(self, zid: int) -> int: ...
    def geom(self, zid: int): ...


@dataclass
class DynamicHolderDeps1D:
    zones: list
    box: float
    cutoff: float
    holder_map: list[int]
    holder_ver: list[int]
    holder_epoch_ref: Callable[[], int]
    max_step_lag: int
    zones_overlapping_range_pbc: Callable

    def deps_table(self, zid: int) -> list[int]:
        z = self.zones[zid]
        deps = self.zones_overlapping_range_pbc(
            z.z0 - self.cutoff, z.z1 + self.cutoff, self.box, self.zones
        )
        return [int(d) for d in deps if int(d) != int(zid)]

    def deps_owner(self, zid: int) -> list[int]:
        z = self.zones[zid]
        b = float(getattr(z, "buffer", 0.0))
        deps = self.zones_overlapping_range_pbc(z.z0 - b, z.z1 + b, self.box, self.zones)
        return [int(d) for d in deps if int(d) != int(zid)]

    def owner_rank(self, zid: int) -> int:
        return int(self.holder_map[int(zid)])

    def geom(self, zid: int) -> ZoneGeom1D:
        z = self.zones[int(zid)]
        return ZoneGeom1D(float(z.z0), float(z.z1))


@dataclass
class StaticRoundRobinOwner:
    size: int

    def owner_rank(self, zid: int) -> int:
        return -1 if int(self.size) <= 0 else (int(zid) % int(self.size))
