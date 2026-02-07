"""
DepsProvider3DBlock module.

This module provides a 3D block decomposition dependencies provider for the
Time Decomposition (TD) molecular dynamics code.

It implements the DepsProvider protocol defined in ``tdmd/deps_provider.py``.
The 3D decomposition splits the simulation domain into a regular grid of
rectangular zones (blocks) along x, y and z axes. Each zone is identified
by a unique integer id derived from its (ix, iy, iz) indices. The zones are
assigned to MPI ranks in a static round-robin fashion.

Dependencies are determined by geometric overlap of the zone's p-neighbourhood
with neighbouring zones within a cutoff radius. For each zone, both
``deps_table`` and ``deps_owner`` return the set of zones whose bounding boxes
overlap the p-neighbourhood of the zone. The owner for a given zone id is
computed statically.

Note: This provider assumes periodic boundary conditions in all three axes.

"""

from typing import List, Tuple
import math


class DepsProvider3DBlock:
    """
    3D block decomposition dependencies provider.

    Parameters
    ----------
    nx : int
        Number of zones along the x dimension.
    ny : int
        Number of zones along the y dimension.
    nz : int
        Number of zones along the z dimension.
    box : Tuple[float, float, float]
        Size of the simulation box (Lx, Ly, Lz).
    cutoff : float
        Interaction cutoff distance.
    mpi_size : int
        Number of MPI ranks.

    The zone id is defined as ``id = ix + iy * nx + iz * nx * ny``.
    The static owner mapping is ``owner_rank(id) = id % mpi_size``.
    """

    def __init__(self, nx: int, ny: int, nz: int,
                 box: Tuple[float, float, float],
                 cutoff: float,
                 mpi_size: int):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.box = box
        self.cutoff = cutoff
        self.mpi_size = mpi_size
        # Compute zone dimensions
        self.zsize_x = box[0] / nx
        self.zsize_y = box[1] / ny
        self.zsize_z = box[2] / nz

    def _index(self, zid: int) -> Tuple[int, int, int]:
        """Convert zone id to (ix, iy, iz) indices."""
        ix = zid % self.nx
        iy = (zid // self.nx) % self.ny
        iz = zid // (self.nx * self.ny)
        return ix, iy, iz

    def _bounds(self, ix: int, iy: int, iz: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Return bounding box of zone indices."""
        x0 = ix * self.zsize_x
        y0 = iy * self.zsize_y
        z0 = iz * self.zsize_z
        x1 = x0 + self.zsize_x
        y1 = y0 + self.zsize_y
        z1 = z0 + self.zsize_z
        return (x0, y0, z0), (x1, y1, z1)

    def owner_rank(self, zid: int) -> int:
        """Return static owner rank for zone id."""
        return zid % self.mpi_size

    def deps_table(self, zid: int) -> List[int]:
        """Return table dependencies for zone zid.

        Dependencies include all zones whose bounding boxes intersect the
        p-neighbourhood of the current zone. The neighbourhood is determined
        by extending the zone extent by the cutoff in each dimension and
        finding all zones within that range using periodic boundary conditions.
        """
        ix, iy, iz = self._index(zid)
        rx = int(math.ceil(self.cutoff / self.zsize_x))
        ry = int(math.ceil(self.cutoff / self.zsize_y))
        rz = int(math.ceil(self.cutoff / self.zsize_z))
        deps: List[int] = []
        seen = set()
        for dx in range(-rx, rx + 1):
            for dy in range(-ry, ry + 1):
                for dz in range(-rz, rz + 1):
                    nix = (ix + dx) % self.nx
                    niy = (iy + dy) % self.ny
                    niz = (iz + dz) % self.nz
                    dep_id = nix + niy * self.nx + niz * self.nx * self.ny
                    if dep_id == int(zid):
                        continue
                    if dep_id in seen:
                        continue
                    seen.add(dep_id)
                    deps.append(dep_id)
        return deps

    def deps_owner(self, zid: int) -> List[int]:
        """Return owner dependencies for zone zid.

        In this provider, owner dependencies are the same as table dependencies
        because any neighbouring zone may receive migrated atoms.
        """
        return self.deps_table(zid)

    def geom(self, zid: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Return bounding box for zone zid."""
        ix, iy, iz = self._index(zid)
        return self._bounds(ix, iy, iz)
