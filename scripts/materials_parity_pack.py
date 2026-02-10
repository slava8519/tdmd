from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import sys

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tdmd.config import load_config
from tdmd.io import load_task, task_to_arrays, validate_task_for_run
from tdmd.observables import compute_observables
from tdmd.potentials import make_potential
from tdmd.serial import run_serial
from tdmd.verify_v2 import run_verify_task


@dataclass
class CaseCheck:
    name: str
    ok: bool
    max_abs_diff: float
    violations: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    property_checks: list[dict[str, Any]] = field(default_factory=list)


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _arr_max_abs(a: Any, b: Any) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.shape != bb.shape:
        return float("inf")
    if aa.size == 0:
        return 0.0
    return float(np.max(np.abs(aa - bb)))


def _scalar_abs(a: Any, b: Any) -> float:
    try:
        return abs(float(a) - float(b))
    except Exception:
        return float("inf")


def _metric_check(name: str, diff: float, tol: float, violations: list[str]) -> None:
    if not np.isfinite(diff) or diff > float(tol):
        violations.append(f"{name}: diff={diff:.6e} tol={float(tol):.6e}")


def _resolve_property_thresholds(
    *, base_thresholds: dict[str, float], fixture: dict[str, Any]
) -> dict[str, float]:
    prop = dict(fixture.get("property_thresholds", {}) or {})
    return {
        "eos_energy_abs": float(prop.get("eos_energy_abs", base_thresholds["energy_abs"])),
        "eos_virial_abs": float(prop.get("eos_virial_abs", base_thresholds["virial_abs"])),
        "eos_press_abs": float(prop.get("eos_press_abs", base_thresholds["press_abs"])),
        "thermo_energy_abs": float(prop.get("thermo_energy_abs", base_thresholds["energy_abs"])),
        "thermo_temp_abs": float(prop.get("thermo_temp_abs", base_thresholds["temp_abs"])),
        "thermo_press_abs": float(prop.get("thermo_press_abs", base_thresholds["press_abs"])),
        "transport_r_abs": float(prop.get("transport_r_abs", base_thresholds["traj_r_abs"])),
        "transport_v_abs": float(prop.get("transport_v_abs", base_thresholds["traj_v_abs"])),
        "transport_verify_abs": float(
            prop.get("transport_verify_abs", base_thresholds["verify_abs"])
        ),
    }


def _property_checks(
    *, actual: dict[str, Any], expected: dict[str, Any], prop_tol: dict[str, float]
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, group: str, diff: float, tol_key: str) -> None:
        tol = float(prop_tol[tol_key])
        checks.append(
            {
                "name": name,
                "group": group,
                "diff": float(diff),
                "tol": tol,
                "ok": bool(np.isfinite(diff) and float(diff) <= tol),
            }
        )

    add(
        "eos.initial.pe",
        "eos",
        _scalar_abs(actual["initial"]["pe"], expected["initial"]["pe"]),
        "eos_energy_abs",
    )
    add(
        "eos.initial.virial",
        "eos",
        _scalar_abs(actual["initial"]["virial"], expected["initial"]["virial"]),
        "eos_virial_abs",
    )
    add(
        "eos.initial.P",
        "eos",
        _scalar_abs(actual["initial"]["P"], expected["initial"]["P"]),
        "eos_press_abs",
    )
    add(
        "eos.final.P",
        "eos",
        _scalar_abs(actual["serial_final"]["P"], expected["serial_final"]["P"]),
        "eos_press_abs",
    )

    add(
        "thermo.initial.T",
        "thermo",
        _scalar_abs(actual["initial"]["T"], expected["initial"]["T"]),
        "thermo_temp_abs",
    )
    add(
        "thermo.final.E",
        "thermo",
        _scalar_abs(actual["serial_final"]["E"], expected["serial_final"]["E"]),
        "thermo_energy_abs",
    )
    add(
        "thermo.final.T",
        "thermo",
        _scalar_abs(actual["serial_final"]["T"], expected["serial_final"]["T"]),
        "thermo_temp_abs",
    )
    add(
        "thermo.final.P",
        "thermo",
        _scalar_abs(actual["serial_final"]["P"], expected["serial_final"]["P"]),
        "thermo_press_abs",
    )

    add(
        "transport.final.positions",
        "transport",
        _arr_max_abs(actual["serial_final"]["positions"], expected["serial_final"]["positions"]),
        "transport_r_abs",
    )
    add(
        "transport.final.velocities",
        "transport",
        _arr_max_abs(actual["serial_final"]["velocities"], expected["serial_final"]["velocities"]),
        "transport_v_abs",
    )
    add(
        "transport.sync.final_dr",
        "transport",
        _scalar_abs(actual["sync_verify"]["final_dr"], expected["sync_verify"]["final_dr"]),
        "transport_verify_abs",
    )
    add(
        "transport.sync.rms_dr",
        "transport",
        _scalar_abs(actual["sync_verify"]["rms_dr"], expected["sync_verify"]["rms_dr"]),
        "transport_verify_abs",
    )
    return checks


def _compute_case_actual(case: dict[str, Any], cfg) -> dict[str, Any]:
    task_path = str(case["task"])
    steps = int(case.get("steps", 5))
    zones_total = int(case.get("zones_total", 1))

    task = load_task(task_path)
    arr = task_to_arrays(task)
    masses = validate_task_for_run(task, allowed_potential_kinds=("eam/alloy",))
    pot = make_potential(task.potential.kind, task.potential.params)

    box = float(task.box.x)
    dt = float(task.dt)
    cutoff = float(task.cutoff)

    f0, pe0, vir0 = pot.forces_energy_virial(
        arr.r.copy(), box=box, cutoff=cutoff, atom_types=arr.atom_types
    )
    obs0 = compute_observables(
        arr.r.copy(),
        arr.v.copy(),
        masses,
        box=box,
        potential=pot,
        cutoff=cutoff,
        atom_types=arr.atom_types,
    )

    r = arr.r.copy()
    v = arr.v.copy()
    run_serial(
        r=r,
        v=v,
        mass=masses,
        box=box,
        potential=pot,
        dt=dt,
        cutoff=cutoff,
        n_steps=steps,
        thermo_every=0,
        atom_types=arr.atom_types,
        device="cpu",
    )
    obsf = compute_observables(
        r, v, masses, box=box, potential=pot, cutoff=cutoff, atom_types=arr.atom_types
    )

    vr = run_verify_task(
        potential=pot,
        r0=arr.r.copy(),
        v0=arr.v.copy(),
        box=box,
        mass=masses,
        dt=dt,
        cutoff=cutoff,
        atom_types=arr.atom_types,
        cell_size=float(cfg.td.cell_size),
        zones_total=zones_total,
        zone_cells_w=int(cfg.td.zone_cells_w),
        zone_cells_s=int(cfg.td.zone_cells_s),
        zone_cells_pattern=cfg.td.zone_cells_pattern,
        traversal=str(cfg.td.traversal),
        buffer_k=float(cfg.td.buffer_k),
        skin_from_buffer=bool(cfg.td.skin_from_buffer),
        use_verlet=False,
        verlet_k_steps=int(cfg.td.verlet_k_steps),
        steps=steps,
        observer_every=1,
        tol_dr=1e-12,
        tol_dv=1e-12,
        tol_dE=1e-12,
        tol_dT=1e-12,
        tol_dP=1e-12,
        decomposition=str(getattr(cfg.td, "decomposition", "1d")),
        zones_nx=int(getattr(cfg.td, "zones_nx", 1)),
        zones_ny=int(getattr(cfg.td, "zones_ny", 1)),
        zones_nz=int(getattr(cfg.td, "zones_nz", 1)),
        sync_mode=True,
        device="cpu",
        strict_min_zone_width=True,
        case_name=str(case["name"]),
    )
    vm = dict(vr.details.get("metrics", {}))

    return {
        "initial": {
            "forces": np.asarray(f0, dtype=float).tolist(),
            "pe": float(pe0),
            "virial": float(vir0),
            "T": float(obs0["T"]),
            "P": float(obs0["P"]),
        },
        "serial_final": {
            "positions": np.asarray(r, dtype=float).tolist(),
            "velocities": np.asarray(v, dtype=float).tolist(),
            "E": float(obsf["E"]),
            "KE": float(obsf["KE"]),
            "PE": float(obsf["PE"]),
            "T": float(obsf["T"]),
            "P": float(obsf["P"]),
        },
        "sync_verify": {
            "max_dr": float(vr.max_dr),
            "max_dv": float(vr.max_dv),
            "max_dE": float(vr.max_dE),
            "max_dT": float(vr.max_dT),
            "max_dP": float(vr.max_dP),
            "final_dr": float(vm.get("final_dr", 0.0)),
            "final_dv": float(vm.get("final_dv", 0.0)),
            "final_dE": float(vm.get("final_dE", 0.0)),
            "final_dT": float(vm.get("final_dT", 0.0)),
            "final_dP": float(vm.get("final_dP", 0.0)),
            "rms_dr": float(vm.get("rms_dr", 0.0)),
            "rms_dv": float(vm.get("rms_dv", 0.0)),
            "rms_dE": float(vm.get("rms_dE", 0.0)),
            "rms_dT": float(vm.get("rms_dT", 0.0)),
            "rms_dP": float(vm.get("rms_dP", 0.0)),
        },
    }


def _check_case(
    case: dict[str, Any],
    actual: dict[str, Any],
    thresholds: dict[str, float],
    prop_tol: dict[str, float],
) -> CaseCheck:
    exp = dict(case.get("expected", {}))
    name = str(case.get("name", "unknown_case"))
    violations: list[str] = []
    max_abs_diff = 0.0

    d = _arr_max_abs(actual["initial"]["forces"], exp["initial"]["forces"])
    max_abs_diff = max(max_abs_diff, d)
    _metric_check("initial.forces", d, float(thresholds["force_abs"]), violations)

    d = _scalar_abs(actual["initial"]["pe"], exp["initial"]["pe"])
    max_abs_diff = max(max_abs_diff, d)
    _metric_check("initial.pe", d, float(thresholds["energy_abs"]), violations)

    d = _scalar_abs(actual["initial"]["virial"], exp["initial"]["virial"])
    max_abs_diff = max(max_abs_diff, d)
    _metric_check("initial.virial", d, float(thresholds["virial_abs"]), violations)

    d = _scalar_abs(actual["initial"]["T"], exp["initial"]["T"])
    max_abs_diff = max(max_abs_diff, d)
    _metric_check("initial.T", d, float(thresholds["temp_abs"]), violations)

    d = _scalar_abs(actual["initial"]["P"], exp["initial"]["P"])
    max_abs_diff = max(max_abs_diff, d)
    _metric_check("initial.P", d, float(thresholds["press_abs"]), violations)

    d = _arr_max_abs(actual["serial_final"]["positions"], exp["serial_final"]["positions"])
    max_abs_diff = max(max_abs_diff, d)
    _metric_check("serial_final.positions", d, float(thresholds["traj_r_abs"]), violations)

    d = _arr_max_abs(actual["serial_final"]["velocities"], exp["serial_final"]["velocities"])
    max_abs_diff = max(max_abs_diff, d)
    _metric_check("serial_final.velocities", d, float(thresholds["traj_v_abs"]), violations)

    for key in ("E", "KE", "PE"):
        d = _scalar_abs(actual["serial_final"][key], exp["serial_final"][key])
        max_abs_diff = max(max_abs_diff, d)
        _metric_check(f"serial_final.{key}", d, float(thresholds["energy_abs"]), violations)
    for key in ("T", "P"):
        d = _scalar_abs(actual["serial_final"][key], exp["serial_final"][key])
        max_abs_diff = max(max_abs_diff, d)
        tol = float(thresholds["temp_abs"] if key == "T" else thresholds["press_abs"])
        _metric_check(f"serial_final.{key}", d, tol, violations)

    sync_keys = (
        "max_dr",
        "max_dv",
        "max_dE",
        "max_dT",
        "max_dP",
        "final_dr",
        "final_dv",
        "final_dE",
        "final_dT",
        "final_dP",
        "rms_dr",
        "rms_dv",
        "rms_dE",
        "rms_dT",
        "rms_dP",
    )
    for key in sync_keys:
        d = _scalar_abs(actual["sync_verify"][key], exp["sync_verify"][key])
        max_abs_diff = max(max_abs_diff, d)
        _metric_check(f"sync_verify.{key}", d, float(thresholds["verify_abs"]), violations)

    prop_checks = _property_checks(actual=actual, expected=exp, prop_tol=prop_tol)
    for pc in prop_checks:
        diff = float(pc["diff"])
        max_abs_diff = max(max_abs_diff, diff)
        if not bool(pc["ok"]):
            violations.append(f"property:{pc['name']}: diff={diff:.6e} tol={float(pc['tol']):.6e}")

    return CaseCheck(
        name=name,
        ok=(len(violations) == 0),
        max_abs_diff=float(max_abs_diff),
        violations=violations,
        details={"actual": actual, "expected": exp},
        property_checks=prop_checks,
    )


def run_suite(*, fixture_path: str, cfg_path: str) -> dict[str, Any]:
    fixture = _load_json(fixture_path)
    cfg = load_config(cfg_path)

    thresholds = dict(fixture.get("thresholds", {}))
    required_thr = (
        "force_abs",
        "energy_abs",
        "virial_abs",
        "temp_abs",
        "press_abs",
        "traj_r_abs",
        "traj_v_abs",
        "verify_abs",
    )
    for key in required_thr:
        if key not in thresholds:
            raise ValueError(f"fixture thresholds missing key: {key}")
    prop_tol = _resolve_property_thresholds(base_thresholds=thresholds, fixture=fixture)

    checks: list[CaseCheck] = []
    for case in list(fixture.get("cases", [])):
        actual = _compute_case_actual(case, cfg)
        checks.append(_check_case(case, actual, thresholds, prop_tol))

    by_property: dict[str, dict[str, Any]] = {}
    for c in checks:
        for pc in list(c.property_checks):
            grp = str(pc.get("group", "unknown"))
            slot = by_property.setdefault(
                grp, {"total": 0, "ok": 0, "fail": 0, "max_abs_diff": 0.0}
            )
            slot["total"] = int(slot["total"]) + 1
            if bool(pc.get("ok", False)):
                slot["ok"] = int(slot["ok"]) + 1
            else:
                slot["fail"] = int(slot["fail"]) + 1
            slot["max_abs_diff"] = float(
                max(float(slot["max_abs_diff"]), float(pc.get("diff", 0.0)))
            )

    total = int(len(checks))
    ok_n = int(sum(1 for c in checks if c.ok))
    summary = {
        "suite_version": int(fixture.get("suite_version", 0)),
        "fixture": str(fixture_path),
        "config": str(cfg_path),
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total": total,
        "ok": ok_n,
        "fail": int(total - ok_n),
        "ok_all": bool(ok_n == total),
        "cases": [
            {
                "name": c.name,
                "ok": bool(c.ok),
                "max_abs_diff": float(c.max_abs_diff),
                "violations": list(c.violations),
                "property_fail": int(
                    sum(1 for x in c.property_checks if not bool(x.get("ok", False)))
                ),
                "property_checks": list(c.property_checks),
            }
            for c in checks
        ],
        "thresholds": thresholds,
        "property_thresholds": prop_tol,
        "by_property": by_property,
        "provenance": dict(fixture.get("provenance", {}) or {}),
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Materials parity suite (metals/alloys) checker")
    ap.add_argument(
        "--fixture",
        default="examples/interop/materials_parity_suite_v1.json",
        help="Fixture JSON with expected parity data",
    )
    ap.add_argument(
        "--config",
        default="examples/td_1d_morse.yaml",
        help="TD config for td_local sync verification path",
    )
    ap.add_argument(
        "--out",
        default="results/materials_parity_suite_v1_summary.json",
        help="Output summary JSON path",
    )
    ap.add_argument("--strict", action="store_true", help="Return non-zero if any case fails")
    args = ap.parse_args()

    summary = run_suite(fixture_path=str(args.fixture), cfg_path=str(args.config))
    os.makedirs(os.path.dirname(str(args.out)) or ".", exist_ok=True)
    with open(str(args.out), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[materials-parity] fixture={args.fixture} total={summary['total']} "
        f"ok={summary['ok']} fail={summary['fail']} ok_all={summary['ok_all']}",
        flush=True,
    )
    for c in summary["cases"]:
        print(
            f"  - {c['name']}: ok={c['ok']} max_abs_diff={float(c['max_abs_diff']):.3e}",
            flush=True,
        )

    if args.strict and not bool(summary["ok_all"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
