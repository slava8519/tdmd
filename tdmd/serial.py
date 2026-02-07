from __future__ import annotations
from typing import Union
import inspect
import numpy as np
from .celllist import forces_on_targets_celllist
from .backend import resolve_backend
from .forces_gpu import (
    forces_on_targets_celllist_backend,
    forces_on_targets_pair_backend,
    supports_pair_gpu,
)
from .ensembles import apply_ensemble_step, build_ensemble_spec
from .potentials import EAMAlloyPotential

def run_serial(r: np.ndarray, v: np.ndarray, mass: Union[float, np.ndarray], box: float, potential,
               dt: float, cutoff: float, n_steps: int, thermo_every: int = 0,
               observer=None, observer_every: int = 0,
               atom_types: np.ndarray | None = None,
               ensemble_kind: str = "nve",
               thermostat: object | None = None,
               barostat: object | None = None,
               device: str = "cpu"):
    """Эталонный последовательный MD (Velocity-Verlet), силы через cell-list по всем атомам."""
    ids = np.arange(r.shape[0], dtype=np.int32)
    rc = cutoff  # для serial достаточно cutoff cell size
    if np.isscalar(mass):
        m = float(mass)
        if m <= 0.0:
            raise ValueError("mass must be positive")
        masses = None
        inv_m = 1.0 / m
    else:
        masses = np.asarray(mass, dtype=float)
        if masses.ndim != 1 or masses.shape[0] != r.shape[0]:
            raise ValueError("mass array must have shape (N,)")
        if np.any(masses <= 0.0):
            raise ValueError("all masses must be positive")
        inv_m = None
    if atom_types is None:
        atom_types = np.ones(r.shape[0], dtype=np.int32)
    else:
        atom_types = np.asarray(atom_types, dtype=np.int32)
        if atom_types.ndim != 1 or atom_types.shape[0] != r.shape[0]:
            raise ValueError("atom_types must have shape (N,)")
    backend = resolve_backend(device)
    use_gpu_pair = (backend.device == "cuda") and supports_pair_gpu(potential)
    ensemble = build_ensemble_spec(
        kind=ensemble_kind,
        thermostat=thermostat,
        barostat=barostat,
        source="serial",
    )
    observer_accepts_box = False
    if observer is not None:
        try:
            sig = inspect.signature(observer)
            params = list(sig.parameters.values())
            observer_accepts_box = (
                any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
                or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
                or len(params) >= 4
            )
        except (TypeError, ValueError):
            observer_accepts_box = False

    def _emit_observer(step: int) -> None:
        if observer is None or not observer_every:
            return
        if observer_accepts_box:
            observer(int(step), r, v, float(box))
        else:
            observer(int(step), r, v)

    def _forces(rr: np.ndarray) -> np.ndarray:
        if use_gpu_pair:
            if isinstance(potential, EAMAlloyPotential):
                f_gpu2 = forces_on_targets_pair_backend(
                    r=rr,
                    box=box,
                    cutoff=cutoff,
                    potential=potential,
                    target_ids=ids,
                    candidate_ids=ids,
                    atom_types=atom_types,
                    backend=backend,
                )
                if f_gpu2 is not None:
                    return np.asarray(f_gpu2, dtype=float)
            else:
                f_gpu = forces_on_targets_celllist_backend(
                    r=rr,
                    box=box,
                    cutoff=cutoff,
                    rc=rc,
                    potential=potential,
                    target_ids=ids,
                    candidate_ids=ids,
                    atom_types=atom_types,
                    backend=backend,
                )
                if f_gpu is not None:
                    return np.asarray(f_gpu, dtype=float)
                f_gpu2 = forces_on_targets_pair_backend(
                    r=rr,
                    box=box,
                    cutoff=cutoff,
                    potential=potential,
                    target_ids=ids,
                    candidate_ids=ids,
                    atom_types=atom_types,
                    backend=backend,
                )
                if f_gpu2 is not None:
                    return np.asarray(f_gpu2, dtype=float)
        if hasattr(potential, "forces_energy_virial"):
            f, _pe, _w = potential.forces_energy_virial(rr, box, cutoff, atom_types)
            return np.asarray(f, dtype=float)
        return forces_on_targets_celllist(rr, box, potential, cutoff, ids, ids, rc=rc, atom_types=atom_types)

    if observer is not None and observer_every:
        _emit_observer(0)
    for step in range(1, n_steps+1):
        f = _forces(r)
        # VV pos + half vel
        if masses is None:
            v_half = v + 0.5 * dt * f * inv_m
        else:
            v_half = v + 0.5 * dt * f / masses[:, None]
        r[:] = (r + dt*v_half) % box
        v[:] = v_half
        # new forces
        f2 = _forces(r)
        if masses is None:
            v[:] = v + 0.5 * dt * f2 * inv_m
        else:
            v[:] = v + 0.5 * dt * f2 / masses[:, None]
        box, _lam_t, _lam_b = apply_ensemble_step(
            step=step,
            ensemble=ensemble,
            r=r,
            v=v,
            mass=(m if masses is None else masses),
            box=box,
            potential=potential,
            cutoff=cutoff,
            atom_types=atom_types,
            dt=dt,
        )
        if observer is not None and observer_every and (step % observer_every == 0):
            _emit_observer(step)
        if thermo_every and (step % thermo_every == 0):
            if masses is None:
                ke = 0.5 * m * float((v * v).sum())
            else:
                ke = 0.5 * float((masses[:, None] * (v * v)).sum())
            print(f"[serial step={step}] KE={ke:.6f} box={float(box):.6f}", flush=True)
