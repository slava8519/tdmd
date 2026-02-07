from __future__ import annotations

from dataclasses import dataclass
import gzip
import math
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class TrajectoryFrame:
    step: int
    n_atoms: int
    box: tuple[float, float, float]
    columns: tuple[str, ...]
    data: dict[str, np.ndarray]


def _open_text(path: str):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _col_dtype(name: str):
    c = str(name).strip().lower()
    if c in ("id", "type", "ix", "iy", "iz"):
        return np.int64
    return np.float64


def iter_lammpstrj(
    path: str,
    *,
    every: int = 1,
    start_step: int | None = None,
    stop_step: int | None = None,
    required_columns: Iterable[str] | None = None,
):
    stride = max(1, int(every))
    req = None
    if required_columns is not None:
        req = {str(x).strip() for x in required_columns if str(x).strip()}
    frame_idx = 0
    with _open_text(path) as f:
        while True:
            line = f.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                raise ValueError(f"unexpected trajectory token: {line.strip()!r}")
            step_txt = f.readline().strip()
            step = int(step_txt)

            hdr = f.readline().strip()
            if hdr != "ITEM: NUMBER OF ATOMS":
                raise ValueError(f"expected 'ITEM: NUMBER OF ATOMS', got {hdr!r}")
            n_atoms = int(f.readline().strip())

            bline = f.readline().strip()
            if not bline.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"expected 'ITEM: BOX BOUNDS ...', got {bline!r}")
            xlo, xhi = [float(x) for x in f.readline().split()[:2]]
            ylo, yhi = [float(x) for x in f.readline().split()[:2]]
            zlo, zhi = [float(x) for x in f.readline().split()[:2]]
            box = (float(xhi - xlo), float(yhi - ylo), float(zhi - zlo))

            aline = f.readline().strip()
            if not aline.startswith("ITEM: ATOMS"):
                raise ValueError(f"expected 'ITEM: ATOMS ...', got {aline!r}")
            columns = tuple(aline.split()[2:])
            col_to_idx = {c: i for i, c in enumerate(columns)}

            if req is None:
                sel = list(columns)
            else:
                missing = sorted(req - set(columns))
                if missing:
                    raise ValueError(f"required columns are missing: {missing}")
                sel = [c for c in columns if c in req]

            data = {c: np.empty((n_atoms,), dtype=_col_dtype(c)) for c in sel}
            for i in range(n_atoms):
                parts = f.readline().split()
                if len(parts) < len(columns):
                    raise ValueError(
                        f"invalid atom row at step={step}: got {len(parts)} fields, expected {len(columns)}"
                    )
                for c in sel:
                    idx = col_to_idx[c]
                    arr = data[c]
                    if np.issubdtype(arr.dtype, np.integer):
                        arr[i] = int(parts[idx])
                    else:
                        arr[i] = float(parts[idx])

            frame_idx += 1
            if start_step is not None and int(step) < int(start_step):
                continue
            if stop_step is not None and int(step) > int(stop_step):
                continue
            if ((frame_idx - 1) % stride) != 0:
                continue
            yield TrajectoryFrame(
                step=int(step),
                n_atoms=int(n_atoms),
                box=box,
                columns=columns,
                data=data,
            )


def _coords_from_frame(frame: TrajectoryFrame) -> np.ndarray:
    if all(k in frame.data for k in ("xu", "yu", "zu")):
        return np.column_stack([frame.data["xu"], frame.data["yu"], frame.data["zu"]]).astype(float)
    if all(k in frame.data for k in ("x", "y", "z")):
        return np.column_stack([frame.data["x"], frame.data["y"], frame.data["z"]]).astype(float)
    raise ValueError("trajectory frame does not contain coordinate columns")


class BaseVizPlugin:
    name = "base"

    def begin(self, frame: TrajectoryFrame) -> None:
        return

    def process(self, frame: TrajectoryFrame) -> dict[str, float]:
        return {}

    def finalize(self) -> dict[str, Any]:
        return {}


class MobilityPlugin(BaseVizPlugin):
    name = "mobility"

    def __init__(self):
        self._ref: np.ndarray | None = None
        self._last_rms = 0.0
        self._last_max = 0.0

    def begin(self, frame: TrajectoryFrame) -> None:
        self._ref = _coords_from_frame(frame)

    def process(self, frame: TrajectoryFrame) -> dict[str, float]:
        if self._ref is None:
            self.begin(frame)
        cur = _coords_from_frame(frame)
        dr = cur - self._ref
        d = np.linalg.norm(dr, axis=1)
        self._last_rms = float(math.sqrt(float(np.mean(d * d)))) if d.size else 0.0
        self._last_max = float(np.max(d)) if d.size else 0.0
        return {"rms": self._last_rms, "max": self._last_max}

    def finalize(self) -> dict[str, Any]:
        return {"final_rms": self._last_rms, "final_max": self._last_max}


class SpeciesMixingPlugin(BaseVizPlugin):
    name = "species_mixing"

    def __init__(self):
        self._last_mixing = 0.0
        self._last_n_types = 0

    def process(self, frame: TrajectoryFrame) -> dict[str, float]:
        if "type" not in frame.data:
            raise ValueError("species_mixing plugin requires 'type' trajectory column")
        tt = np.asarray(frame.data["type"], dtype=np.int64)
        if tt.size == 0:
            self._last_mixing = 0.0
            self._last_n_types = 0
            return {"index": 0.0, "n_types": 0.0}
        vals, cnt = np.unique(tt, return_counts=True)
        p = cnt.astype(float) / float(tt.size)
        self._last_mixing = float(1.0 - np.sum(p * p))
        self._last_n_types = int(vals.size)
        return {"index": self._last_mixing, "n_types": float(self._last_n_types)}

    def finalize(self) -> dict[str, Any]:
        return {"final_index": self._last_mixing, "final_n_types": self._last_n_types}


class RegionOccupancyPlugin(BaseVizPlugin):
    name = "region_occupancy"

    def __init__(
        self,
        *,
        xlo: float,
        xhi: float,
        ylo: float,
        yhi: float,
        zlo: float,
        zhi: float,
        label: str = "region",
    ):
        self.xlo = float(xlo); self.xhi = float(xhi)
        self.ylo = float(ylo); self.yhi = float(yhi)
        self.zlo = float(zlo); self.zhi = float(zhi)
        self.label = str(label or "region")
        self._ref_count: int | None = None
        self._last_count = 0
        self._last_frac = 0.0
        self._last_fill = 0.0

    def _mask(self, rr: np.ndarray) -> np.ndarray:
        return (
            (rr[:, 0] >= self.xlo) & (rr[:, 0] <= self.xhi)
            & (rr[:, 1] >= self.ylo) & (rr[:, 1] <= self.yhi)
            & (rr[:, 2] >= self.zlo) & (rr[:, 2] <= self.zhi)
        )

    def begin(self, frame: TrajectoryFrame) -> None:
        rr = _coords_from_frame(frame)
        self._ref_count = int(np.sum(self._mask(rr)))

    def process(self, frame: TrajectoryFrame) -> dict[str, float]:
        if self._ref_count is None:
            self.begin(frame)
        rr = _coords_from_frame(frame)
        m = self._mask(rr)
        self._last_count = int(np.sum(m))
        self._last_frac = float(self._last_count) / float(max(1, rr.shape[0]))
        self._last_fill = float(self._last_count) / float(max(1, int(self._ref_count or 0)))
        return {"count": float(self._last_count), "fraction": self._last_frac, "fill_fraction": self._last_fill}

    def finalize(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "reference_count": int(self._ref_count or 0),
            "final_count": int(self._last_count),
            "final_fraction": float(self._last_frac),
            "final_fill_fraction": float(self._last_fill),
        }


def _parse_kv_params(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    raw = str(text or "").strip()
    if not raw:
        return out
    for tok in raw.split(","):
        t = str(tok).strip()
        if not t:
            continue
        if "=" not in t:
            out[t] = "1"
            continue
        k, v = t.split("=", 1)
        out[str(k).strip()] = str(v).strip()
    return out


def build_plugin(spec: str) -> BaseVizPlugin:
    raw = str(spec or "").strip()
    if not raw:
        raise ValueError("empty plugin spec")
    if ":" in raw:
        name, param_txt = raw.split(":", 1)
    else:
        name, param_txt = raw, ""
    name = str(name).strip().lower()
    params = _parse_kv_params(param_txt)
    if name in ("mobility", "displacement"):
        return MobilityPlugin()
    if name in ("species_mixing", "mixing"):
        return SpeciesMixingPlugin()
    if name in ("region", "region_occupancy", "crack_fill"):
        keys = ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi")
        miss = [k for k in keys if k not in params]
        if miss:
            raise ValueError(
                f"region plugin requires params {keys}; missing {miss}. "
                "example: region:xlo=0,xhi=10,ylo=0,yhi=10,zlo=0,zhi=10"
            )
        return RegionOccupancyPlugin(
            xlo=float(params["xlo"]),
            xhi=float(params["xhi"]),
            ylo=float(params["ylo"]),
            yhi=float(params["yhi"]),
            zlo=float(params["zlo"]),
            zhi=float(params["zhi"]),
            label=str(params.get("label", "region")),
        )
    raise ValueError(f"unknown plugin: {name}")


def run_plugins(
    traj_path: str,
    plugins: list[BaseVizPlugin],
    *,
    every: int = 1,
    start_step: int | None = None,
    stop_step: int | None = None,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    rows: list[dict[str, float]] = []
    begun = False
    n_frames = 0
    for fr in iter_lammpstrj(
        traj_path,
        every=every,
        start_step=start_step,
        stop_step=stop_step,
        required_columns=None,
    ):
        if not begun:
            for p in plugins:
                p.begin(fr)
            begun = True
        row: dict[str, float] = {"step": float(fr.step)}
        for p in plugins:
            vals = p.process(fr)
            for k, v in vals.items():
                row[f"{p.name}.{k}"] = float(v)
        rows.append(row)
        n_frames += 1
    summary = {
        "frames": int(n_frames),
        "plugins": {p.name: p.finalize() for p in plugins},
    }
    return rows, summary
