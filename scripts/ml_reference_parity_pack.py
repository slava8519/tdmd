from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import sys

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tdmd.config import load_config
from tdmd.io import load_task, task_to_arrays, validate_task_for_run
from tdmd.observables import compute_observables
from tdmd.potentials import describe_ml_reference_contract, make_potential
from tdmd.serial import run_serial
from tdmd.verify_v2 import run_verify_task


@dataclass
class CaseCheck:
    name: str
    ok: bool
    max_abs_diff: float
    violations: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
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


def _compute_case_actual(case: dict[str, Any], cfg) -> dict[str, Any]:
    task_path = str(case["task"])
    steps = int(case.get("steps", 4))
    zones_total = int(case.get("zones_total", 1))

    task = load_task(task_path)
    arr = task_to_arrays(task)
    masses = validate_task_for_run(task, allowed_potential_kinds=("ml/reference",))
    pot = make_potential(task.potential.kind, task.potential.params)
    contract = describe_ml_reference_contract(pot)

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
        "contract": contract,
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
            "max_dr": float(vm.get("max_dr", 0.0)),
            "max_dv": float(vm.get("max_dv", 0.0)),
            "max_dE": float(vm.get("max_dE", 0.0)),
            "max_dT": float(vm.get("max_dT", 0.0)),
            "max_dP": float(vm.get("max_dP", 0.0)),
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


def _compare_case(case: dict[str, Any], actual: dict[str, Any], thr: dict[str, float]) -> CaseCheck:
    expected = dict(case.get("expected", {}) or {})
    violations: list[str] = []

    contract_actual = dict(actual.get("contract", {}) or {})
    contract_expected = dict(expected.get("contract", {}) or {})
    if contract_actual != contract_expected:
        violations.append("contract mismatch")

    init_a = dict(actual["initial"])
    init_e = dict(expected["initial"])
    _metric_check(
        "initial.forces",
        _arr_max_abs(init_a["forces"], init_e["forces"]),
        thr["force_abs"],
        violations,
    )
    _metric_check(
        "initial.pe", _scalar_abs(init_a["pe"], init_e["pe"]), thr["energy_abs"], violations
    )
    _metric_check(
        "initial.virial",
        _scalar_abs(init_a["virial"], init_e["virial"]),
        thr["virial_abs"],
        violations,
    )
    _metric_check("initial.T", _scalar_abs(init_a["T"], init_e["T"]), thr["temp_abs"], violations)
    _metric_check("initial.P", _scalar_abs(init_a["P"], init_e["P"]), thr["press_abs"], violations)

    final_a = dict(actual["serial_final"])
    final_e = dict(expected["serial_final"])
    _metric_check(
        "serial_final.positions",
        _arr_max_abs(final_a["positions"], final_e["positions"]),
        thr["traj_r_abs"],
        violations,
    )
    _metric_check(
        "serial_final.velocities",
        _arr_max_abs(final_a["velocities"], final_e["velocities"]),
        thr["traj_v_abs"],
        violations,
    )
    _metric_check(
        "serial_final.E", _scalar_abs(final_a["E"], final_e["E"]), thr["energy_abs"], violations
    )
    _metric_check(
        "serial_final.KE", _scalar_abs(final_a["KE"], final_e["KE"]), thr["energy_abs"], violations
    )
    _metric_check(
        "serial_final.PE", _scalar_abs(final_a["PE"], final_e["PE"]), thr["energy_abs"], violations
    )
    _metric_check(
        "serial_final.T", _scalar_abs(final_a["T"], final_e["T"]), thr["temp_abs"], violations
    )
    _metric_check(
        "serial_final.P", _scalar_abs(final_a["P"], final_e["P"]), thr["press_abs"], violations
    )

    verify_a = dict(actual["sync_verify"])
    verify_e = dict(expected["sync_verify"])
    for key in (
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
    ):
        _metric_check(
            f"sync_verify.{key}",
            _scalar_abs(verify_a[key], verify_e[key]),
            thr["verify_abs"],
            violations,
        )

    diffs = [
        _arr_max_abs(init_a["forces"], init_e["forces"]),
        _scalar_abs(init_a["pe"], init_e["pe"]),
        _scalar_abs(init_a["virial"], init_e["virial"]),
        _arr_max_abs(final_a["positions"], final_e["positions"]),
        _arr_max_abs(final_a["velocities"], final_e["velocities"]),
    ]
    max_abs_diff = max(float(x) for x in diffs)
    return CaseCheck(
        name=str(case["name"]),
        ok=not violations,
        max_abs_diff=max_abs_diff,
        violations=violations,
        details={"expected": expected, "actual": actual},
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Strict parity pack for ml/reference fixture suite")
    ap.add_argument(
        "--fixture",
        default="examples/interop/ml_reference_suite_v1.json",
        help="fixture JSON path",
    )
    ap.add_argument("--config", default="examples/td_1d_morse.yaml", help="TDMD config YAML")
    ap.add_argument("--out", default="", help="optional output JSON summary path")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    fixture = _load_json(str(args.fixture))
    if int(fixture.get("suite_version", 0)) != 1:
        raise SystemExit("unsupported ml_reference suite_version")
    cases = list(fixture.get("cases", []) or [])
    if not cases:
        raise SystemExit("ml_reference fixture has no cases")

    thr_raw = dict(fixture.get("thresholds", {}) or {})
    thr = {
        "force_abs": float(thr_raw.get("force_abs", 1e-12)),
        "energy_abs": float(thr_raw.get("energy_abs", 1e-12)),
        "virial_abs": float(thr_raw.get("virial_abs", 1e-12)),
        "temp_abs": float(thr_raw.get("temp_abs", 1e-12)),
        "press_abs": float(thr_raw.get("press_abs", 1e-12)),
        "traj_r_abs": float(thr_raw.get("traj_r_abs", 1e-12)),
        "traj_v_abs": float(thr_raw.get("traj_v_abs", 1e-12)),
        "verify_abs": float(thr_raw.get("verify_abs", 1e-12)),
    }

    cfg = load_config(str(args.config))
    checks: list[CaseCheck] = []
    for case in cases:
        actual = _compute_case_actual(case, cfg)
        checks.append(_compare_case(case, actual, thr))

    ok_all = all(bool(c.ok) for c in checks)
    summary = {
        "suite_version": int(fixture.get("suite_version", 1)),
        "family": str(fixture.get("family", "ml/reference:quadratic_density")),
        "description": str(fixture.get("description", "")),
        "thresholds": thr,
        "config": str(args.config),
        "fixture": str(args.fixture),
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total": len(checks),
        "ok": int(sum(1 for c in checks if c.ok)),
        "fail": int(sum(1 for c in checks if not c.ok)),
        "ok_all": bool(ok_all),
        "cases": [
            {
                "name": c.name,
                "ok": bool(c.ok),
                "max_abs_diff": float(c.max_abs_diff),
                "violations": list(c.violations),
                "details": dict(c.details),
            }
            for c in checks
        ],
    }

    if args.out:
        out_path = str(args.out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(
        f"[ml-reference-parity] fixture={args.fixture} total={summary['total']} ok={summary['ok']} "
        f"fail={summary['fail']} ok_all={summary['ok_all']}"
    )
    for case in summary["cases"]:
        print(f"  - {case['name']}: ok={case['ok']} max_abs_diff={float(case['max_abs_diff']):.3e}")

    if bool(args.strict) and not bool(summary["ok_all"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
