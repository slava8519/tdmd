from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

from .observables import compute_observables
from .state import kinetic_energy, temperature_from_ke


@dataclass(frozen=True)
class EnsembleControl:
    kind: str
    params: dict[str, Any]


@dataclass(frozen=True)
class EnsembleSpec:
    kind: str
    thermostat: Optional[EnsembleControl] = None
    barostat: Optional[EnsembleControl] = None


def _as_control(obj: Any, key: str) -> Optional[EnsembleControl]:
    if obj is None:
        return None
    if isinstance(obj, EnsembleControl):
        return obj
    kind = getattr(obj, "kind", None)
    params = getattr(obj, "params", None)
    if kind is None and isinstance(obj, dict):
        kind = obj.get("kind")
        params = obj.get("params", {})
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError(f"{key}.kind must be a non-empty string")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError(f"{key}.params must be a mapping")
    return EnsembleControl(kind=str(kind).strip().lower(), params=dict(params))


def build_ensemble_spec(
    *,
    kind: str,
    thermostat: Any = None,
    barostat: Any = None,
    source: str = "runtime",
) -> EnsembleSpec:
    k = str(kind).strip().lower()
    if k not in ("nve", "nvt", "npt"):
        raise ValueError(f"{source} ensemble.kind must be one of ['npt', 'nve', 'nvt']")
    th = _as_control(thermostat, f"{source}.ensemble.thermostat")
    br = _as_control(barostat, f"{source}.ensemble.barostat")
    if k == "nve":
        if th is not None or br is not None:
            raise ValueError(f"{source} ensemble.kind=nve must not include thermostat/barostat")
    elif k == "nvt":
        if th is None:
            raise ValueError(f"{source} ensemble.kind=nvt requires thermostat")
        if br is not None:
            raise ValueError(f"{source} ensemble.kind=nvt must not include barostat")
    elif k == "npt":
        if th is None:
            raise ValueError(f"{source} ensemble.kind=npt requires thermostat")
        if br is None:
            raise ValueError(f"{source} ensemble.kind=npt requires barostat")
    return EnsembleSpec(kind=k, thermostat=th, barostat=br)


def _param_float(
    params: dict[str, Any], key: str, default: Optional[float] = None, *, positive: bool = False
) -> float:
    if key not in params:
        if default is None:
            raise ValueError(f"missing required parameter: {key}")
        out = float(default)
    else:
        out = float(params[key])
    if positive and out <= 0.0:
        raise ValueError(f"parameter '{key}' must be positive")
    return out


def _param_int(params: dict[str, Any], key: str, default: int) -> int:
    out = int(params.get(key, default))
    if out < 1:
        raise ValueError(f"parameter '{key}' must be >= 1")
    return out


def _apply_berendsen_thermostat(
    *,
    v: np.ndarray,
    mass: Union[float, np.ndarray],
    dt: float,
    params: dict[str, Any],
) -> float:
    t_target = _param_float(params, "t_target", positive=True)
    tau = _param_float(params, "tau", positive=True)
    max_scale_step = _param_float(params, "max_scale_step", 0.20, positive=True)

    ke = kinetic_energy(v, mass)
    t_inst = temperature_from_ke(ke, int(v.shape[0]))
    if t_inst <= 1e-20:
        return 1.0

    lam2 = 1.0 + (float(dt) / tau) * ((t_target / t_inst) - 1.0)
    lam2 = max(lam2, 1e-12)
    lam = math.sqrt(lam2)
    lo = max(1e-6, 1.0 - max_scale_step)
    hi = 1.0 + max_scale_step
    lam = float(np.clip(lam, lo, hi))
    v *= lam
    return lam


def _apply_berendsen_barostat(
    *,
    r: np.ndarray,
    v: np.ndarray,
    box: float,
    mass: Union[float, np.ndarray],
    potential: Any,
    cutoff: float,
    atom_types: Optional[np.ndarray],
    dt: float,
    params: dict[str, Any],
) -> tuple[float, float]:
    p_target = _param_float(params, "p_target")
    tau = _param_float(params, "tau", positive=True)
    compressibility = _param_float(params, "compressibility", 1.0e-4, positive=True)
    max_volume_scale_step = _param_float(params, "max_volume_scale_step", 0.10, positive=True)
    scale_velocities = bool(params.get("scale_velocities", True))

    obs = compute_observables(
        r, v, mass, float(box), potential, float(cutoff), atom_types=atom_types
    )
    p_inst = float(obs.get("P", 0.0))

    # Berendsen isotropic volume coupling:
    # V' = V * (1 + dt/tau * beta * (P_inst - P_target)).
    vol_scale = 1.0 + (float(dt) / tau) * compressibility * (p_inst - p_target)
    lo = max(1e-8, 1.0 - max_volume_scale_step)
    hi = 1.0 + max_volume_scale_step
    vol_scale = float(np.clip(vol_scale, lo, hi))
    vol_scale = max(vol_scale, 1e-8)

    lin = float(vol_scale ** (1.0 / 3.0))
    new_box = float(box) * lin
    if new_box <= 0.0:
        raise ValueError("barostat produced non-positive box size")
    r[:] = (r * lin) % new_box
    if scale_velocities:
        v *= lin
    return new_box, lin


def apply_ensemble_step(
    *,
    step: int,
    ensemble: EnsembleSpec,
    r: np.ndarray,
    v: np.ndarray,
    mass: Union[float, np.ndarray],
    box: float,
    potential: Any,
    cutoff: float,
    atom_types: Optional[np.ndarray],
    dt: float,
) -> tuple[float, float, float]:
    """Apply ensemble controls once at step boundary.

    Returns: (new_box, lambda_t, lambda_box_linear)
    """
    if ensemble.kind == "nve":
        return float(box), 1.0, 1.0

    lam_t = 1.0
    lam_box = 1.0
    out_box = float(box)

    if ensemble.thermostat is not None:
        th_kind = str(ensemble.thermostat.kind).strip().lower()
        th_every = _param_int(ensemble.thermostat.params, "every", 1)
        if (int(step) % th_every) == 0:
            if th_kind in ("berendsen", "ber"):
                lam_t = _apply_berendsen_thermostat(
                    v=v, mass=mass, dt=float(dt), params=ensemble.thermostat.params
                )
            else:
                raise ValueError(f"unsupported thermostat kind: {th_kind}")

    if ensemble.kind == "npt":
        if ensemble.barostat is None:
            raise ValueError("ensemble.kind=npt requires barostat")
        br_kind = str(ensemble.barostat.kind).strip().lower()
        br_every = _param_int(ensemble.barostat.params, "every", 1)
        if (int(step) % br_every) == 0:
            if br_kind in ("berendsen", "ber"):
                out_box, lam_box = _apply_berendsen_barostat(
                    r=r,
                    v=v,
                    box=float(out_box),
                    mass=mass,
                    potential=potential,
                    cutoff=float(cutoff),
                    atom_types=atom_types,
                    dt=float(dt),
                    params=ensemble.barostat.params,
                )
            else:
                raise ValueError(f"unsupported barostat kind: {br_kind}")

    return float(out_box), float(lam_t), float(lam_box)
