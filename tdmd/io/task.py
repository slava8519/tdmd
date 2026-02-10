from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import yaml

from ..potentials import canonical_potential_kind, ensure_pair_coeffs_complete, parse_pair_coeffs

_LAMMPS_UNITS = {
    "lj",
    "real",
    "metal",
    "si",
    "cgs",
    "electron",
    "micro",
    "nano",
}
_ENSEMBLE_KINDS = {"nve", "nvt", "npt"}
_RUNTIME_POTENTIAL_PARAMS = {
    "lj": {"epsilon", "sigma", "pair_coeffs"},
    "morse": {"D_e", "a", "r0", "pair_coeffs"},
    "table": {"file", "keyword", "style"},
    "eam/alloy": {"file", "elements"},
}


class TaskValidationError(ValueError):
    pass


@dataclass(frozen=True)
class TaskBox:
    x: float
    y: float
    z: float
    pbc: tuple[bool, bool, bool]


@dataclass(frozen=True)
class TaskAtom:
    id: int
    type: int
    mass: float
    charge: Optional[float]
    r: tuple[float, float, float]
    v: tuple[float, float, float]


@dataclass(frozen=True)
class TaskPotential:
    kind: str
    params: dict[str, Any]


@dataclass(frozen=True)
class TaskThermostat:
    kind: str
    params: dict[str, Any]


@dataclass(frozen=True)
class TaskBarostat:
    kind: str
    params: dict[str, Any]


@dataclass(frozen=True)
class TaskEnsemble:
    kind: str
    thermostat: Optional[TaskThermostat] = None
    barostat: Optional[TaskBarostat] = None


@dataclass(frozen=True)
class Task:
    task_version: int
    units: str
    box: TaskBox
    atoms: list[TaskAtom]
    potential: TaskPotential
    cutoff: float
    dt: float
    steps: int
    ensemble: TaskEnsemble
    # Legacy alias kept for backward compatibility with existing task files.
    thermostat: Optional[TaskThermostat] = None


@dataclass(frozen=True)
class TaskArrays:
    r: np.ndarray
    v: np.ndarray
    atom_ids: np.ndarray
    atom_types: np.ndarray
    masses: np.ndarray
    charges: np.ndarray


def _err(msg: str) -> TaskValidationError:
    return TaskValidationError(msg)


def _expect_dict(d: Any, key: str) -> dict:
    if not isinstance(d, dict):
        raise _err(f"{key} must be a mapping")
    return d


def _expect_seq(x: Any, key: str, n: int | None = None) -> Sequence[Any]:
    if not isinstance(x, (list, tuple)):
        raise _err(f"{key} must be a list")
    if n is not None and len(x) != n:
        raise _err(f"{key} must have length {n}")
    return x


def _expect_str(x: Any, key: str) -> str:
    if not isinstance(x, str):
        raise _err(f"{key} must be a string")
    return x


def _expect_int(x: Any, key: str) -> int:
    if not isinstance(x, (int, np.integer)):
        raise _err(f"{key} must be an int")
    return int(x)


def _expect_float(x: Any, key: str) -> float:
    if not isinstance(x, (int, float, np.integer, np.floating)):
        raise _err(f"{key} must be a number")
    return float(x)


def _expect_bool(x: Any, key: str) -> bool:
    if isinstance(x, bool):
        return bool(x)
    if isinstance(x, (int, np.integer)) and x in (0, 1):
        return bool(x)
    raise _err(f"{key} must be a bool")


def _parse_box(d: dict) -> TaskBox:
    box = _expect_dict(d.get("box", None), "box")
    x = _expect_float(box.get("x", None), "box.x")
    y = _expect_float(box.get("y", None), "box.y")
    z = _expect_float(box.get("z", None), "box.z")
    if x <= 0 or y <= 0 or z <= 0:
        raise _err("box dimensions must be positive")
    pbc = box.get("pbc", [True, True, True])
    pbc = _expect_seq(pbc, "box.pbc", 3)
    pbc_b = tuple(_expect_bool(pbc[i], f"box.pbc[{i}]") for i in range(3))
    return TaskBox(x=float(x), y=float(y), z=float(z), pbc=pbc_b)


def _parse_atoms(d: dict) -> list[TaskAtom]:
    atoms_raw = d.get("atoms", None)
    if atoms_raw is None:
        raise _err("atoms list is required")
    atoms_raw = _expect_seq(atoms_raw, "atoms")
    atoms: list[TaskAtom] = []
    seen_ids = set()
    for i, a in enumerate(atoms_raw):
        if not isinstance(a, dict):
            raise _err(f"atoms[{i}] must be a mapping")
        aid = _expect_int(a.get("id", None), f"atoms[{i}].id")
        if aid in seen_ids:
            raise _err(f"duplicate atom id: {aid}")
        seen_ids.add(aid)
        atype = _expect_int(a.get("type", None), f"atoms[{i}].type")
        mass = _expect_float(a.get("mass", None), f"atoms[{i}].mass")
        charge = a.get("charge", None)
        if charge is not None:
            charge = _expect_float(charge, f"atoms[{i}].charge")
        r = _expect_seq(a.get("r", None), f"atoms[{i}].r", 3)
        v = _expect_seq(a.get("v", None), f"atoms[{i}].v", 3)
        r3 = tuple(_expect_float(r[j], f"atoms[{i}].r[{j}]") for j in range(3))
        v3 = tuple(_expect_float(v[j], f"atoms[{i}].v[{j}]") for j in range(3))
        atoms.append(TaskAtom(id=aid, type=atype, mass=mass, charge=charge, r=r3, v=v3))
    if not atoms:
        raise _err("atoms list must be non-empty")
    return atoms


def _parse_potential(d: dict, atom_types: np.ndarray) -> TaskPotential:
    pot = _expect_dict(d.get("potential", None), "potential")
    kind = canonical_potential_kind(_expect_str(pot.get("kind", None), "potential.kind"))
    params = pot.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise _err("potential.params must be a mapping")
    if kind not in ("lj", "morse", "table", "eam/alloy"):
        raise _err(f"unsupported potential.kind: {kind}")
    if kind == "table":
        if "file" not in params or "keyword" not in params:
            raise _err("potential.params for table must include 'file' and 'keyword'")
    if kind == "eam/alloy":
        if "file" not in params:
            raise _err("potential.params for eam/alloy must include 'file'")
        elems = params.get("elements", None)
        if not isinstance(elems, (list, tuple)) or len(elems) == 0:
            raise _err("potential.params for eam/alloy must include non-empty 'elements' list")
        max_type = int(np.max(atom_types)) if atom_types.size else 0
        if max_type > len(elems):
            raise _err(
                f"eam/alloy elements list length ({len(elems)}) is smaller than max atom type ({max_type})"
            )
        for i, nm in enumerate(elems):
            if not isinstance(nm, str) or not nm.strip():
                raise _err(f"potential.params.elements[{i}] must be a non-empty string")
    if kind in ("lj", "morse"):
        try:
            pair_coeffs = parse_pair_coeffs(kind, params)
            ensure_pair_coeffs_complete(pair_coeffs, atom_types)
        except ValueError as exc:
            raise _err(str(exc)) from exc
    return TaskPotential(kind=kind, params=dict(params))


def _parse_thermostat_block(data: Any, key: str) -> TaskThermostat:
    th = _expect_dict(data, key)
    extra = sorted(set(th.keys()) - {"kind", "params"})
    if extra:
        raise _err(f"{key} contains unsupported keys: {extra}")
    kind = _expect_str(th.get("kind", None), f"{key}.kind").strip().lower()
    if not kind:
        raise _err(f"{key}.kind must be non-empty")
    params = th.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise _err(f"{key}.params must be a mapping")
    return TaskThermostat(kind=kind, params=dict(params))


def _parse_barostat_block(data: Any, key: str) -> TaskBarostat:
    br = _expect_dict(data, key)
    extra = sorted(set(br.keys()) - {"kind", "params"})
    if extra:
        raise _err(f"{key} contains unsupported keys: {extra}")
    kind = _expect_str(br.get("kind", None), f"{key}.kind").strip().lower()
    if not kind:
        raise _err(f"{key}.kind must be non-empty")
    params = br.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise _err(f"{key}.params must be a mapping")
    return TaskBarostat(kind=kind, params=dict(params))


def _parse_thermostat(d: dict) -> Optional[TaskThermostat]:
    th = d.get("thermostat", None)
    if th is None:
        return None
    return _parse_thermostat_block(th, "thermostat")


def _parse_ensemble(d: dict, legacy_thermostat: Optional[TaskThermostat]) -> TaskEnsemble:
    ens = d.get("ensemble", None)
    if ens is None:
        if legacy_thermostat is not None:
            # Legacy top-level thermostat implies NVT intent.
            return TaskEnsemble(kind="nvt", thermostat=legacy_thermostat, barostat=None)
        return TaskEnsemble(kind="nve", thermostat=None, barostat=None)

    if legacy_thermostat is not None:
        raise _err("top-level thermostat is deprecated: use ensemble.thermostat only")

    ens = _expect_dict(ens, "ensemble")
    extra = sorted(set(ens.keys()) - {"kind", "thermostat", "barostat"})
    if extra:
        raise _err(f"ensemble contains unsupported keys: {extra}")

    kind = _expect_str(ens.get("kind", None), "ensemble.kind").strip().lower()
    if kind not in _ENSEMBLE_KINDS:
        raise _err(f"ensemble.kind must be one of {sorted(_ENSEMBLE_KINDS)}")
    thermostat = None
    barostat = None
    if ens.get("thermostat", None) is not None:
        thermostat = _parse_thermostat_block(ens.get("thermostat"), "ensemble.thermostat")
    if ens.get("barostat", None) is not None:
        barostat = _parse_barostat_block(ens.get("barostat"), "ensemble.barostat")

    if kind == "nve":
        if thermostat is not None or barostat is not None:
            raise _err("ensemble.kind=nve must not include thermostat/barostat")
    elif kind == "nvt":
        if thermostat is None:
            raise _err("ensemble.kind=nvt requires ensemble.thermostat")
        if barostat is not None:
            raise _err("ensemble.kind=nvt must not include ensemble.barostat")
    elif kind == "npt":
        if thermostat is None:
            raise _err("ensemble.kind=npt requires ensemble.thermostat")
        if barostat is None:
            raise _err("ensemble.kind=npt requires ensemble.barostat")
    return TaskEnsemble(kind=kind, thermostat=thermostat, barostat=barostat)


def parse_task_dict(d: dict) -> Task:
    if not isinstance(d, dict):
        raise _err("task root must be a mapping")
    task_version = _expect_int(d.get("task_version", None), "task_version")
    if task_version != 1:
        raise _err(f"unsupported task_version: {task_version}")
    units = _expect_str(d.get("units", None), "units").lower()
    if units not in _LAMMPS_UNITS:
        raise _err(f"units must be one of {sorted(_LAMMPS_UNITS)}")
    box = _parse_box(d)
    atoms = _parse_atoms(d)
    atom_types = np.asarray([int(a.type) for a in atoms], dtype=np.int32)
    potential = _parse_potential(d, atom_types)
    cutoff = _expect_float(d.get("cutoff", None), "cutoff")
    dt = _expect_float(d.get("dt", None), "dt")
    steps = _expect_int(d.get("steps", None), "steps")
    if cutoff <= 0.0:
        raise _err("cutoff must be positive")
    if dt <= 0.0:
        raise _err("dt must be positive")
    if steps <= 0:
        raise _err("steps must be positive")
    thermostat = _parse_thermostat(d)
    ensemble = _parse_ensemble(d, thermostat)
    return Task(
        task_version=task_version,
        units=units,
        box=box,
        atoms=atoms,
        potential=potential,
        cutoff=cutoff,
        dt=dt,
        steps=steps,
        ensemble=ensemble,
        thermostat=thermostat,
    )


def load_task(path: str) -> Task:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return parse_task_dict(data)


def task_to_arrays(task: Task) -> TaskArrays:
    atoms = sorted(task.atoms, key=lambda a: a.id)
    ids = np.array([a.id for a in atoms], dtype=np.int32)
    types = np.array([a.type for a in atoms], dtype=np.int32)
    masses = np.array([a.mass for a in atoms], dtype=float)
    charges = np.array([0.0 if a.charge is None else float(a.charge) for a in atoms], dtype=float)
    r = np.array([a.r for a in atoms], dtype=float)
    v = np.array([a.v for a in atoms], dtype=float)
    return TaskArrays(r=r, v=v, atom_ids=ids, atom_types=types, masses=masses, charges=charges)


def validate_task_for_run(
    task: Task,
    *,
    require_uniform_mass: bool = False,
    require_single_type: bool = False,
    allowed_potential_kinds: tuple[str, ...] = ("lj", "morse", "table"),
    allowed_ensemble_kinds: tuple[str, ...] = ("nve",),
) -> np.ndarray:
    if task.thermostat is not None:
        raise _err(
            "thermostat is not supported in TDMD run yet (legacy field parsed for interop only)"
        )
    allowed_ensembles = tuple(str(k).strip().lower() for k in allowed_ensemble_kinds)
    if task.ensemble.kind not in allowed_ensembles:
        raise _err(
            f"ensemble.kind '{task.ensemble.kind}' not supported for TDMD run "
            f"(allowed: {sorted(set(allowed_ensembles))})"
        )
    if not all(bool(x) for x in task.box.pbc):
        raise _err(
            "TDMD run requires periodic boundaries in all directions (box.pbc must be [true,true,true])"
        )
    if any(a.charge is not None for a in task.atoms):
        raise _err("atom charges are not supported in TDMD run yet (charges are interop-only)")
    allowed = tuple(canonical_potential_kind(k) for k in allowed_potential_kinds)
    if task.potential.kind not in allowed:
        raise _err(f"potential.kind '{task.potential.kind}' not supported for TDMD run")
    allowed_params = _RUNTIME_POTENTIAL_PARAMS.get(task.potential.kind, set())
    unknown = sorted(set(task.potential.params.keys()) - set(allowed_params))
    if unknown:
        raise _err(
            f"unsupported potential.params for '{task.potential.kind}' in TDMD run: {unknown}; "
            f"allowed: {sorted(allowed_params)}"
        )
    if task.units != "lj":
        # The runtime computes in the numeric values as provided by the task and does not perform
        # unit conversion; make this explicit to avoid schema/runtime ambiguity.
        warnings.warn(
            f"TDMD run performs no unit conversion (units='{task.units}'): values are used as-is",
            RuntimeWarning,
        )
    arr = task_to_arrays(task)
    masses = np.asarray(arr.masses, dtype=float)
    if masses.ndim != 1 or masses.shape[0] != len(task.atoms):
        raise _err("internal error: invalid masses array shape from task")
    if np.any(masses <= 0.0):
        raise _err("all atom masses must be positive")
    if require_uniform_mass and (np.unique(masses).size != 1):
        raise _err("TDMD run in this mode requires a single mass for all atoms")
    types = np.asarray(arr.atom_types, dtype=np.int32)
    if require_single_type and (np.unique(types).size != 1):
        raise _err("TDMD run in this mode requires a single atom type")
    # TDMD uses cubic box
    if abs(task.box.x - task.box.y) > 1e-12 or abs(task.box.x - task.box.z) > 1e-12:
        raise _err("TDMD run requires cubic box (x==y==z)")
    return masses
