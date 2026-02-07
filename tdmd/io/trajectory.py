from __future__ import annotations
import gzip
import os
import numpy as np

from .manifest import trajectory_manifest_payload, write_manifest
from .snapshot_forces import compute_forces_snapshot


class TrajectoryWriter:
    """LAMMPS dump (lammpstrj) writer."""
    SCHEMA_NAME = "tdmd.trajectory.lammps_dump"
    SCHEMA_VERSION = 1

    def __init__(
        self,
        path: str,
        *,
        box: tuple[float, float, float],
        pbc: tuple[bool, bool, bool],
        atom_ids: np.ndarray,
        atom_types: np.ndarray,
        channels: tuple[str, ...] | list[str] | None = None,
        compression: str = "none",
        write_output_manifest: bool = True,
        force_potential=None,
        force_cutoff: float | None = None,
        force_atom_types: np.ndarray | None = None,
    ):
        self.path = path
        self.box = (float(box[0]), float(box[1]), float(box[2]))
        self.pbc = tuple(bool(x) for x in pbc)
        self.atom_ids = atom_ids.astype(np.int64)
        self.atom_types = atom_types.astype(np.int32)
        self._n_atoms = int(self.atom_ids.size)

        raw_channels = tuple(channels or ())
        norm: list[str] = []
        for ch in raw_channels:
            c = str(ch).strip().lower()
            if not c or c in ("basic",):
                continue
            if c == "all":
                for x in ("unwrapped", "image", "force"):
                    if x not in norm:
                        norm.append(x)
                continue
            if c not in ("unwrapped", "image", "force"):
                raise ValueError(
                    f"unsupported trajectory channel: {ch!r}; "
                    "allowed: basic,all,unwrapped,image,force"
                )
            if c not in norm:
                norm.append(c)
        self._channels = tuple(norm)
        self._include_unwrapped = ("unwrapped" in self._channels)
        self._include_image = ("image" in self._channels)
        self._include_force = ("force" in self._channels)

        self._compression = str(compression).strip().lower() or "none"
        if self._compression not in ("none", "gz"):
            raise ValueError("compression must be one of: none, gz")

        self._force_potential = force_potential
        self._force_cutoff = None if force_cutoff is None else float(force_cutoff)
        self._force_atom_types = None if force_atom_types is None else np.asarray(force_atom_types, dtype=np.int32)
        if self._include_force:
            if self._force_potential is None or self._force_cutoff is None:
                raise ValueError(
                    "trajectory force channel requires force_potential and force_cutoff"
                )

        self._images = np.zeros((self._n_atoms, 3), dtype=np.int64)
        self._prev_r: np.ndarray | None = None

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if self._compression == "gz":
            self._f = gzip.open(path, "wt", encoding="utf-8")
        else:
            self._f = open(path, "w", encoding="utf-8")

        if write_output_manifest:
            mpath = f"{self.path}.manifest.json"
            mp = trajectory_manifest_payload(
                path=self.path,
                format_name=self.SCHEMA_NAME,
                schema_version=self.SCHEMA_VERSION,
                columns=self._atom_columns(),
                n_atoms=self._n_atoms,
                pbc=self.pbc,
                box=self.box,
                channels={
                    "unwrapped": bool(self._include_unwrapped),
                    "image": bool(self._include_image),
                    "force": bool(self._include_force),
                },
                compression=self._compression,
            )
            write_manifest(mpath, mp)

    def _atom_columns(self) -> list[str]:
        cols = ["id", "type", "x", "y", "z", "vx", "vy", "vz"]
        if self._include_unwrapped:
            cols.extend(["xu", "yu", "zu"])
        if self._include_image:
            cols.extend(["ix", "iy", "iz"])
        if self._include_force:
            cols.extend(["fx", "fy", "fz"])
        return cols

    def _update_images(self, r: np.ndarray, box_xyz: tuple[float, float, float]) -> None:
        if self._prev_r is None:
            self._prev_r = np.asarray(r, dtype=float).copy()
            return
        cur = np.asarray(r, dtype=float)
        prev = self._prev_r
        if cur.shape != prev.shape:
            raise ValueError("trajectory frame atom count changed; unsupported")
        box_arr = np.asarray(box_xyz, dtype=float)[None, :]
        dr = cur - prev
        half = 0.5 * box_arr
        self._images[dr < -half] += 1
        self._images[dr > half] -= 1
        self._prev_r = cur.copy()

    def _forces_for_frame(self, r: np.ndarray, box_xyz: tuple[float, float, float]) -> np.ndarray:
        # Force snapshots currently support cubic boxes only, matching runtime contract.
        if abs(float(box_xyz[0]) - float(box_xyz[1])) > 1e-12 or abs(float(box_xyz[0]) - float(box_xyz[2])) > 1e-12:
            raise ValueError("trajectory force channel requires cubic box")
        return compute_forces_snapshot(
            r=np.asarray(r, dtype=float),
            box=float(box_xyz[0]),
            potential=self._force_potential,
            cutoff=float(self._force_cutoff),
            atom_types=self._force_atom_types,
        )

    def write(
        self,
        step: int,
        r: np.ndarray,
        v: np.ndarray,
        box_value: tuple[float, float, float] | None = None,
    ):
        n = int(r.shape[0])
        if n != self._n_atoms:
            raise ValueError(
                f"trajectory frame atom count mismatch: got {n}, expected {self._n_atoms}"
            )
        xlo, ylo, zlo = 0.0, 0.0, 0.0
        if box_value is None:
            xhi, yhi, zhi = self.box
        else:
            xhi = float(box_value[0]); yhi = float(box_value[1]); zhi = float(box_value[2])
        box_xyz = (xhi, yhi, zhi)
        bflags = ["pp" if b else "ff" for b in self.pbc]

        if self._include_unwrapped or self._include_image:
            self._update_images(r, box_xyz=box_xyz)
        ff = None
        if self._include_force:
            ff = self._forces_for_frame(r, box_xyz=box_xyz)

        f = self._f
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{int(step)}\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{n}\n")
        f.write(f"ITEM: BOX BOUNDS {bflags[0]} {bflags[1]} {bflags[2]}\n")
        f.write(f"{xlo:.8f} {xhi:.8f}\n")
        f.write(f"{ylo:.8f} {yhi:.8f}\n")
        f.write(f"{zlo:.8f} {zhi:.8f}\n")
        f.write("ITEM: ATOMS " + " ".join(self._atom_columns()) + "\n")
        ids = self.atom_ids
        types = self.atom_types
        if self._include_unwrapped:
            unwrapped = np.asarray(r, dtype=float) + (self._images.astype(float) * np.asarray(box_xyz, dtype=float)[None, :])
        else:
            unwrapped = None
        for i in range(n):
            row = [
                f"{int(ids[i])}",
                f"{int(types[i])}",
                f"{r[i,0]:.8f}",
                f"{r[i,1]:.8f}",
                f"{r[i,2]:.8f}",
                f"{v[i,0]:.8f}",
                f"{v[i,1]:.8f}",
                f"{v[i,2]:.8f}",
            ]
            if self._include_unwrapped and unwrapped is not None:
                row.extend(
                    [
                        f"{unwrapped[i,0]:.8f}",
                        f"{unwrapped[i,1]:.8f}",
                        f"{unwrapped[i,2]:.8f}",
                    ]
                )
            if self._include_image:
                row.extend(
                    [
                        str(int(self._images[i, 0])),
                        str(int(self._images[i, 1])),
                        str(int(self._images[i, 2])),
                    ]
                )
            if self._include_force and ff is not None:
                row.extend(
                    [
                        f"{ff[i,0]:.8f}",
                        f"{ff[i,1]:.8f}",
                        f"{ff[i,2]:.8f}",
                    ]
                )
            f.write(" ".join(row) + "\n")
        f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass
