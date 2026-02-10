from __future__ import annotations

from typing import Union

import numpy as np

from .atoms import normalize_atom_types, normalize_mass
from .backend import resolve_backend
from .celllist import forces_on_targets_celllist
from .ensembles import apply_ensemble_step, build_ensemble_spec
from .force_dispatch import try_gpu_forces_on_targets
from .observer import emit_observer, observer_accepts_box


def run_serial(
    r: np.ndarray,
    v: np.ndarray,
    mass: Union[float, np.ndarray],
    box: float,
    potential,
    dt: float,
    cutoff: float,
    n_steps: int,
    thermo_every: int = 0,
    observer=None,
    observer_every: int = 0,
    atom_types: np.ndarray | None = None,
    ensemble_kind: str = "nve",
    thermostat: object | None = None,
    barostat: object | None = None,
    device: str = "cpu",
):
    """Эталонный последовательный MD (Velocity-Verlet), силы через cell-list по всем атомам."""
    ids = np.arange(r.shape[0], dtype=np.int32)
    rc = cutoff  # для serial достаточно cutoff cell size

    mass_scalar, mass_arr, inv_mass_scalar = normalize_mass(mass, n_atoms=r.shape[0])
    if mass_arr is None:
        m = float(mass_scalar)
        masses = None
        inv_m = float(inv_mass_scalar)
    else:
        m = None
        masses = mass_arr
        inv_m = None

    atom_types = normalize_atom_types(atom_types, n_atoms=r.shape[0])
    backend = resolve_backend(device)
    ensemble = build_ensemble_spec(
        kind=ensemble_kind,
        thermostat=thermostat,
        barostat=barostat,
        source="serial",
    )
    accepts_box = observer_accepts_box(observer)

    def _emit_observer(step: int) -> None:
        if not observer_every:
            return
        emit_observer(observer, accepts_box=accepts_box, step=step, r=r, v=v, box=float(box))

    def _forces(rr: np.ndarray) -> np.ndarray:
        f_gpu = try_gpu_forces_on_targets(
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
            return f_gpu
        if hasattr(potential, "forces_energy_virial"):
            f, _pe, _w = potential.forces_energy_virial(rr, box, cutoff, atom_types)
            return np.asarray(f, dtype=float)
        return forces_on_targets_celllist(
            rr, box, potential, cutoff, ids, ids, rc=rc, atom_types=atom_types
        )

    if observer is not None and observer_every:
        _emit_observer(0)
    for step in range(1, n_steps + 1):
        f = _forces(r)
        # VV pos + half vel
        if masses is None:
            v_half = v + 0.5 * dt * f * inv_m
        else:
            v_half = v + 0.5 * dt * f / masses[:, None]
        r[:] = (r + dt * v_half) % box
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
