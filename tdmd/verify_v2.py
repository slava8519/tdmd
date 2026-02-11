from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Union

import numpy as np

from .observables import compute_observables
from .run_configs import VerifyTaskRunConfig
from .testcases import default_cases, make_case_state


@dataclass
class VerifyResult:
    case: str
    steps: int
    max_dr: float
    max_dv: float
    max_dE: float
    max_dT: float
    max_dP: float
    ok: bool
    details: dict


def _details_payload(
    *,
    common,
    mapA,
    mapB,
    max_dr: float,
    max_dv: float,
    max_dE: float,
    max_dT: float,
    max_dP: float,
    final_dr: float,
    final_dv: float,
    final_dE: float,
    final_dT: float,
    final_dP: float,
    rms_dr: float,
    rms_dv: float,
    rms_dE: float,
    rms_dT: float,
    rms_dP: float,
    tol_dr: float,
    tol_dv: float,
    tol_dE: float,
    tol_dT: float,
    tol_dP: float,
    invariants: dict | None = None,
) -> dict:
    metric_ok = {
        "dr": bool(max_dr <= tol_dr),
        "dv": bool(max_dv <= tol_dv),
        "dE": bool(max_dE <= tol_dE),
        "dT": bool(max_dT <= tol_dT),
        "dP": bool(max_dP <= tol_dP),
    }
    violations = [k for k, ok in metric_ok.items() if not ok]
    invariants = dict(invariants or {})
    for k, v in invariants.items():
        if int(v) != 0:
            violations.append(f"{k}={int(v)}")
    return {
        "common_steps": common,
        "serial_last": mapA[common[-1]] if common else None,
        "td_last": mapB[common[-1]] if common else None,
        "metrics": {
            "final_dr": final_dr,
            "final_dv": final_dv,
            "final_dE": final_dE,
            "final_dT": final_dT,
            "final_dP": final_dP,
            "rms_dr": rms_dr,
            "rms_dv": rms_dv,
            "rms_dE": rms_dE,
            "rms_dT": rms_dT,
            "rms_dP": rms_dP,
        },
        "metric_ok": metric_ok,
        "tol": {"dr": tol_dr, "dv": tol_dv, "dE": tol_dE, "dT": tol_dT, "dP": tol_dP},
        "invariants": invariants,
        "violations": violations,
    }


def run_verify_v2(
    *,
    potential,
    mass: float,
    dt: float,
    cutoff: float,
    cell_size: float,
    zones_total: int,
    zone_cells_w: int,
    zone_cells_s: int,
    zone_cells_pattern,
    traversal: str,
    buffer_k: float,
    skin_from_buffer: bool,
    use_verlet: bool,
    verlet_k_steps: int,
    steps: int = 50,
    observer_every: int = 5,
    tol_dr: float = 1e-6,
    tol_dv: float = 1e-6,
    tol_dE: float = 1e-5,
    tol_dT: float = 1e-5,
    tol_dP: float = 1e-4,
    cases=None,
    decomposition: str = "1d",
    zones_nx: int = 1,
    zones_ny: int = 1,
    zones_nz: int = 1,
    sync_mode: bool = False,
    device: str = "cpu",
    strict_min_zone_width: bool = False,
    ensemble_kind: str = "nve",
    thermostat: object | None = None,
    barostat: object | None = None,
    chaos_mode: bool = False,
    chaos_seed: int = 12345,
    chaos_delay_prob: float = 0.0,
):
    from .serial import run_serial
    from .td_local import run_td_local

    if cases is None:
        cases = default_cases()

    results: list[VerifyResult] = []

    for case in cases:
        r0, v0, box = make_case_state(case, mass)

        # records
        obsA = []
        obsB = []
        posA = {}
        dr_time: list[float] = []
        dv_time: list[float] = []

        def obsA_cb(step, r, v, box_cur=None):
            box_use = float(box if box_cur is None else box_cur)
            obsA.append((int(step), compute_observables(r, v, mass, box_use, potential, cutoff)))
            posA[int(step)] = (r.copy(), v.copy())

        def obsB_cb(step, r, v, box_cur=None):
            box_use = float(box if box_cur is None else box_cur)
            obsB.append((int(step), compute_observables(r, v, mass, box_use, potential, cutoff)))
            key = int(step)
            if key in posA:
                rA_step, vA_step = posA[key]
                dr_step = np.sqrt(((rA_step - r) ** 2).sum(axis=1))
                dv_step = np.sqrt(((vA_step - v) ** 2).sum(axis=1))
                dr_time.append(float(dr_step.max()) if dr_step.size else 0.0)
                dv_time.append(float(dv_step.max()) if dv_step.size else 0.0)

        rA = r0.copy()
        vA = v0.copy()
        rB = r0.copy()
        vB = v0.copy()

        run_serial(
            rA,
            vA,
            mass,
            box,
            potential,
            dt,
            cutoff,
            steps,
            thermo_every=0,
            observer=obsA_cb,
            observer_every=observer_every,
            ensemble_kind=ensemble_kind,
            thermostat=thermostat,
            barostat=barostat,
            device=device,
        )

        run_td_local(
            rB,
            vB,
            mass,
            box,
            potential,
            dt,
            cutoff,
            steps,
            observer=obsB_cb,
            observer_every=observer_every,
            cell_size=cell_size,
            zones_total=zones_total,
            zone_cells_w=zone_cells_w,
            zone_cells_s=zone_cells_s,
            zone_cells_pattern=zone_cells_pattern,
            traversal=traversal,
            buffer_k=buffer_k,
            skin_from_buffer=skin_from_buffer,
            use_verlet=use_verlet,
            verlet_k_steps=verlet_k_steps,
            chaos_mode=bool(chaos_mode),
            chaos_seed=int(chaos_seed),
            chaos_delay_prob=float(chaos_delay_prob),
            decomposition=str(decomposition),
            zones_nx=int(zones_nx),
            zones_ny=int(zones_ny),
            zones_nz=int(zones_nz),
            sync_mode=bool(sync_mode),
            ensemble_kind=ensemble_kind,
            thermostat=thermostat,
            barostat=barostat,
            device=device,
            strict_min_zone_width=bool(strict_min_zone_width),
        )

        dr = np.sqrt(((rA - rB) ** 2).sum(axis=1))
        dv = np.sqrt(((vA - vB) ** 2).sum(axis=1))
        max_dr = float(dr.max()) if dr.size else 0.0
        max_dv = float(dv.max()) if dv.size else 0.0

        # align time series by step
        mapA = {s: o for s, o in obsA}
        mapB = {s: o for s, o in obsB}
        common = sorted(set(mapA.keys()).intersection(mapB.keys()))
        dE = []
        dT = []
        dP = []
        for s in common:
            dE.append(abs(mapA[s]["E"] - mapB[s]["E"]))
            dT.append(abs(mapA[s]["T"] - mapB[s]["T"]))
            dP.append(abs(mapA[s]["P"] - mapB[s]["P"]))
        max_dE = float(max(dE)) if dE else 0.0
        max_dT = float(max(dT)) if dT else 0.0
        max_dP = float(max(dP)) if dP else 0.0
        final_dE = float(dE[-1]) if dE else 0.0
        final_dT = float(dT[-1]) if dT else 0.0
        final_dP = float(dP[-1]) if dP else 0.0
        rms_dE = float(np.sqrt(np.mean(np.square(dE)))) if dE else 0.0
        rms_dT = float(np.sqrt(np.mean(np.square(dT)))) if dT else 0.0
        rms_dP = float(np.sqrt(np.mean(np.square(dP)))) if dP else 0.0

        ok = (
            max_dr <= tol_dr
            and max_dv <= tol_dv
            and max_dE <= tol_dE
            and max_dT <= tol_dT
            and max_dP <= tol_dP
        )

        invariants = {"hG3": 0, "hV3": 0, "tG3": 0}
        final_dr = float(dr_time[-1]) if dr_time else float(max_dr)
        final_dv = float(dv_time[-1]) if dv_time else float(max_dv)
        rms_dr = float(np.sqrt(np.mean(np.square(dr_time)))) if dr_time else 0.0
        rms_dv = float(np.sqrt(np.mean(np.square(dv_time)))) if dv_time else 0.0
        details = _details_payload(
            common=common,
            mapA=mapA,
            mapB=mapB,
            max_dr=max_dr,
            max_dv=max_dv,
            max_dE=max_dE,
            max_dT=max_dT,
            max_dP=max_dP,
            final_dr=final_dr,
            final_dv=final_dv,
            final_dE=final_dE,
            final_dT=final_dT,
            final_dP=final_dP,
            rms_dr=rms_dr,
            rms_dv=rms_dv,
            rms_dE=rms_dE,
            rms_dT=rms_dT,
            rms_dP=rms_dP,
            tol_dr=tol_dr,
            tol_dv=tol_dv,
            tol_dE=tol_dE,
            tol_dT=tol_dT,
            tol_dP=tol_dP,
            invariants=invariants,
        )

        results.append(
            VerifyResult(
                case=case.name,
                steps=int(steps),
                max_dr=max_dr,
                max_dv=max_dv,
                max_dE=max_dE,
                max_dT=max_dT,
                max_dP=max_dP,
                ok=bool(ok),
                details=details,
            )
        )

    return results


def _rand_state(
    n_atoms: int, box: float, temperature: float, mass: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    r = rng.uniform(0.0, float(box), size=(int(n_atoms), 3)).astype(float)
    # Maxwell-Boltzmann (kB=1)
    sigma = math.sqrt(float(temperature) / float(mass))
    v = rng.normal(0.0, sigma, size=(int(n_atoms), 3)).astype(float)
    v -= v.mean(axis=0, keepdims=True)
    return r, v


def run_verify_config(
    *,
    potential,
    n_atoms: int,
    box: float,
    temperature: float,
    seed: int,
    mass: float,
    dt: float,
    cutoff: float,
    cell_size: float,
    zones_total: int,
    zone_cells_w: int,
    zone_cells_s: int,
    zone_cells_pattern,
    traversal: str,
    buffer_k: float,
    skin_from_buffer: bool,
    use_verlet: bool,
    verlet_k_steps: int,
    steps: int = 50,
    observer_every: int = 5,
    tol_dr: float = 1e-6,
    tol_dv: float = 1e-6,
    tol_dE: float = 1e-5,
    tol_dT: float = 1e-5,
    tol_dP: float = 1e-4,
    decomposition: str = "1d",
    zones_nx: int = 1,
    zones_ny: int = 1,
    zones_nz: int = 1,
    sync_mode: bool = False,
    device: str = "cpu",
    strict_min_zone_width: bool = False,
    ensemble_kind: str = "nve",
    thermostat: object | None = None,
    barostat: object | None = None,
    chaos_mode: bool = False,
    chaos_seed: int = 12345,
    chaos_delay_prob: float = 0.0,
) -> VerifyResult:
    """Verify serial vs TD-local on a configuration-defined random system.

    This is the Codex/CI-friendly reference because it uses the project's YAML system settings
    and avoids special crystal/gas constructors that may be tuned for LJ-only experiments.
    """
    r0, v0 = _rand_state(int(n_atoms), float(box), float(temperature), float(mass), int(seed))
    # reuse run_verify_v2 machinery by creating a single synthetic case name
    from .serial import run_serial
    from .td_local import run_td_local

    rA = r0.copy()
    vA = v0.copy()
    obsA = []
    posA = {}
    dr_time: list[float] = []
    dv_time: list[float] = []

    def obsA_cb(step, r, v, box_cur=None):
        box_use = float(box if box_cur is None else box_cur)
        obsA.append((step, compute_observables(r, v, mass, box_use, potential, cutoff)))
        posA[int(step)] = (r.copy(), v.copy())

    run_serial(
        rA,
        vA,
        mass,
        box,
        potential,
        dt,
        cutoff,
        steps,
        observer=obsA_cb,
        observer_every=observer_every,
        ensemble_kind=ensemble_kind,
        thermostat=thermostat,
        barostat=barostat,
        device=device,
    )

    rB = r0.copy()
    vB = v0.copy()
    obsB = []

    def obsB_cb(step, r, v, box_cur=None):
        box_use = float(box if box_cur is None else box_cur)
        obsB.append((step, compute_observables(r, v, mass, box_use, potential, cutoff)))
        key = int(step)
        if key in posA:
            rA_step, vA_step = posA[key]
            dr_step = np.sqrt(((rA_step - r) ** 2).sum(axis=1))
            dv_step = np.sqrt(((vA_step - v) ** 2).sum(axis=1))
            dr_time.append(float(dr_step.max()) if dr_step.size else 0.0)
            dv_time.append(float(dv_step.max()) if dv_step.size else 0.0)

    run_td_local(
        rB,
        vB,
        mass,
        box,
        potential,
        dt,
        cutoff,
        steps,
        observer=obsB_cb,
        observer_every=observer_every,
        cell_size=cell_size,
        zones_total=zones_total,
        zone_cells_w=zone_cells_w,
        zone_cells_s=zone_cells_s,
        zone_cells_pattern=zone_cells_pattern,
        traversal=traversal,
        buffer_k=buffer_k,
        skin_from_buffer=skin_from_buffer,
        use_verlet=use_verlet,
        verlet_k_steps=verlet_k_steps,
        chaos_mode=bool(chaos_mode),
        chaos_seed=int(chaos_seed),
        chaos_delay_prob=float(chaos_delay_prob),
        decomposition=str(decomposition),
        zones_nx=int(zones_nx),
        zones_ny=int(zones_ny),
        zones_nz=int(zones_nz),
        sync_mode=bool(sync_mode),
        ensemble_kind=ensemble_kind,
        thermostat=thermostat,
        barostat=barostat,
        device=device,
        strict_min_zone_width=bool(strict_min_zone_width),
    )

    dr = float(np.sqrt(((rA - rB) ** 2).sum(axis=1)).max())
    dv = float(np.sqrt(((vA - vB) ** 2).sum(axis=1)).max())

    stepsA = {int(s): o for (s, o) in obsA}
    stepsB = {int(s): o for (s, o) in obsB}
    common = sorted(set(stepsA.keys()) & set(stepsB.keys()))
    dE = [abs(stepsA[s]["E"] - stepsB[s]["E"]) for s in common]
    dT = [abs(stepsA[s]["T"] - stepsB[s]["T"]) for s in common]
    dP = [abs(stepsA[s]["P"] - stepsB[s]["P"]) for s in common]
    max_dE = max(dE) if dE else float("inf")
    max_dT = max(dT) if dT else float("inf")
    max_dP = max(dP) if dP else float("inf")
    final_dE = float(dE[-1]) if dE else float("inf")
    final_dT = float(dT[-1]) if dT else float("inf")
    final_dP = float(dP[-1]) if dP else float("inf")
    rms_dE = float(np.sqrt(np.mean(np.square(dE)))) if dE else float("inf")
    rms_dT = float(np.sqrt(np.mean(np.square(dT)))) if dT else float("inf")
    rms_dP = float(np.sqrt(np.mean(np.square(dP)))) if dP else float("inf")

    ok = (
        (dr <= tol_dr)
        and (dv <= tol_dv)
        and (max_dE <= tol_dE)
        and (max_dT <= tol_dT)
        and (max_dP <= tol_dP)
    )

    invariants = {"hG3": 0, "hV3": 0, "tG3": 0}
    final_dr = float(dr_time[-1]) if dr_time else float(dr)
    final_dv = float(dv_time[-1]) if dv_time else float(dv)
    dr_all = np.sqrt(((rA - rB) ** 2).sum(axis=1))
    dv_all = np.sqrt(((vA - vB) ** 2).sum(axis=1))
    rms_dr = (
        float(np.sqrt(np.mean(np.square(dr_time))))
        if dr_time
        else (float(np.sqrt(np.mean(np.square(dr_all)))) if dr_all.size else 0.0)
    )
    rms_dv = (
        float(np.sqrt(np.mean(np.square(dv_time))))
        if dv_time
        else (float(np.sqrt(np.mean(np.square(dv_all)))) if dv_all.size else 0.0)
    )
    details = _details_payload(
        common=common,
        mapA=stepsA,
        mapB=stepsB,
        max_dr=dr,
        max_dv=dv,
        max_dE=float(max_dE),
        max_dT=float(max_dT),
        max_dP=float(max_dP),
        final_dr=final_dr,
        final_dv=final_dv,
        final_dE=final_dE,
        final_dT=final_dT,
        final_dP=final_dP,
        rms_dr=rms_dr,
        rms_dv=rms_dv,
        rms_dE=rms_dE,
        rms_dT=rms_dT,
        rms_dP=rms_dP,
        tol_dr=tol_dr,
        tol_dv=tol_dv,
        tol_dE=tol_dE,
        tol_dT=tol_dT,
        tol_dP=tol_dP,
        invariants=invariants,
    )
    return VerifyResult(
        case="cfg_system",
        steps=int(steps),
        max_dr=dr,
        max_dv=dv,
        max_dE=float(max_dE),
        max_dT=float(max_dT),
        max_dP=float(max_dP),
        ok=bool(ok),
        details=details,
    )


def _run_verify_task_legacy(
    *,
    potential,
    r0: np.ndarray,
    v0: np.ndarray,
    box: float,
    mass: Union[float, np.ndarray],
    dt: float,
    cutoff: float,
    atom_types: np.ndarray | None = None,
    cell_size: float,
    zones_total: int,
    zone_cells_w: int,
    zone_cells_s: int,
    zone_cells_pattern,
    traversal: str,
    buffer_k: float,
    skin_from_buffer: bool,
    use_verlet: bool,
    verlet_k_steps: int,
    steps: int = 50,
    observer_every: int = 5,
    tol_dr: float = 1e-6,
    tol_dv: float = 1e-6,
    tol_dE: float = 1e-5,
    tol_dT: float = 1e-5,
    tol_dP: float = 1e-4,
    decomposition: str = "1d",
    zones_nx: int = 1,
    zones_ny: int = 1,
    zones_nz: int = 1,
    sync_mode: bool = False,
    device: str = "cpu",
    strict_min_zone_width: bool = False,
    ensemble_kind: str = "nve",
    thermostat: object | None = None,
    barostat: object | None = None,
    chaos_mode: bool = False,
    chaos_seed: int = 12345,
    chaos_delay_prob: float = 0.0,
    case_name: str = "interop_task",
) -> VerifyResult:
    """Verify serial vs TD-local on an explicit task-defined initial state."""
    from .serial import run_serial
    from .td_local import run_td_local

    obsA = []
    posA = {}
    obsB = []
    dr_time: list[float] = []
    dv_time: list[float] = []

    def obsA_cb(step, r, v, box_cur=None):
        box_use = float(box if box_cur is None else box_cur)
        obsA.append(
            (
                int(step),
                compute_observables(r, v, mass, box_use, potential, cutoff, atom_types=atom_types),
            )
        )
        posA[int(step)] = (r.copy(), v.copy())

    def obsB_cb(step, r, v, box_cur=None):
        box_use = float(box if box_cur is None else box_cur)
        obsB.append(
            (
                int(step),
                compute_observables(r, v, mass, box_use, potential, cutoff, atom_types=atom_types),
            )
        )
        key = int(step)
        if key in posA:
            rA_step, vA_step = posA[key]
            dr_step = np.sqrt(((rA_step - r) ** 2).sum(axis=1))
            dv_step = np.sqrt(((vA_step - v) ** 2).sum(axis=1))
            dr_time.append(float(dr_step.max()) if dr_step.size else 0.0)
            dv_time.append(float(dv_step.max()) if dv_step.size else 0.0)

    rA = r0.copy()
    vA = v0.copy()
    rB = r0.copy()
    vB = v0.copy()

    run_serial(
        rA,
        vA,
        mass,
        box,
        potential,
        dt,
        cutoff,
        steps,
        thermo_every=0,
        observer=obsA_cb,
        observer_every=observer_every,
        atom_types=atom_types,
        ensemble_kind=ensemble_kind,
        thermostat=thermostat,
        barostat=barostat,
        device=device,
    )

    run_td_local(
        rB,
        vB,
        mass,
        box,
        potential,
        dt,
        cutoff,
        steps,
        observer=obsB_cb,
        observer_every=observer_every,
        atom_types=atom_types,
        cell_size=cell_size,
        zones_total=zones_total,
        zone_cells_w=zone_cells_w,
        zone_cells_s=zone_cells_s,
        zone_cells_pattern=zone_cells_pattern,
        traversal=traversal,
        buffer_k=buffer_k,
        skin_from_buffer=skin_from_buffer,
        use_verlet=use_verlet,
        verlet_k_steps=verlet_k_steps,
        chaos_mode=bool(chaos_mode),
        chaos_seed=int(chaos_seed),
        chaos_delay_prob=float(chaos_delay_prob),
        decomposition=str(decomposition),
        zones_nx=int(zones_nx),
        zones_ny=int(zones_ny),
        zones_nz=int(zones_nz),
        sync_mode=bool(sync_mode),
        ensemble_kind=ensemble_kind,
        thermostat=thermostat,
        barostat=barostat,
        device=device,
        strict_min_zone_width=bool(strict_min_zone_width),
    )

    dr = np.sqrt(((rA - rB) ** 2).sum(axis=1))
    dv = np.sqrt(((vA - vB) ** 2).sum(axis=1))
    max_dr = float(dr.max()) if dr.size else 0.0
    max_dv = float(dv.max()) if dv.size else 0.0

    stepsA = {int(s): o for (s, o) in obsA}
    stepsB = {int(s): o for (s, o) in obsB}
    common = sorted(set(stepsA.keys()) & set(stepsB.keys()))
    dE = [abs(stepsA[s]["E"] - stepsB[s]["E"]) for s in common]
    dT = [abs(stepsA[s]["T"] - stepsB[s]["T"]) for s in common]
    dP = [abs(stepsA[s]["P"] - stepsB[s]["P"]) for s in common]
    max_dE = max(dE) if dE else float("inf")
    max_dT = max(dT) if dT else float("inf")
    max_dP = max(dP) if dP else float("inf")
    final_dE = float(dE[-1]) if dE else float("inf")
    final_dT = float(dT[-1]) if dT else float("inf")
    final_dP = float(dP[-1]) if dP else float("inf")
    rms_dE = float(np.sqrt(np.mean(np.square(dE)))) if dE else float("inf")
    rms_dT = float(np.sqrt(np.mean(np.square(dT)))) if dT else float("inf")
    rms_dP = float(np.sqrt(np.mean(np.square(dP)))) if dP else float("inf")

    ok = (
        (max_dr <= tol_dr)
        and (max_dv <= tol_dv)
        and (max_dE <= tol_dE)
        and (max_dT <= tol_dT)
        and (max_dP <= tol_dP)
    )

    invariants = {"hG3": 0, "hV3": 0, "tG3": 0}
    final_dr = float(dr_time[-1]) if dr_time else float(max_dr)
    final_dv = float(dv_time[-1]) if dv_time else float(max_dv)
    rms_dr = (
        float(np.sqrt(np.mean(np.square(dr_time))))
        if dr_time
        else (float(np.sqrt(np.mean(np.square(dr)))) if dr.size else 0.0)
    )
    rms_dv = (
        float(np.sqrt(np.mean(np.square(dv_time))))
        if dv_time
        else (float(np.sqrt(np.mean(np.square(dv)))) if dv.size else 0.0)
    )
    details = _details_payload(
        common=common,
        mapA=stepsA,
        mapB=stepsB,
        max_dr=max_dr,
        max_dv=max_dv,
        max_dE=float(max_dE),
        max_dT=float(max_dT),
        max_dP=float(max_dP),
        final_dr=final_dr,
        final_dv=final_dv,
        final_dE=final_dE,
        final_dT=final_dT,
        final_dP=final_dP,
        rms_dr=rms_dr,
        rms_dv=rms_dv,
        rms_dE=rms_dE,
        rms_dT=rms_dT,
        rms_dP=rms_dP,
        tol_dr=tol_dr,
        tol_dv=tol_dv,
        tol_dE=tol_dE,
        tol_dT=tol_dT,
        tol_dP=tol_dP,
        invariants=invariants,
    )
    return VerifyResult(
        case=case_name,
        steps=int(steps),
        max_dr=max_dr,
        max_dv=max_dv,
        max_dE=float(max_dE),
        max_dT=float(max_dT),
        max_dP=float(max_dP),
        ok=bool(ok),
        details=details,
    )


def run_verify_task(
    *,
    potential,
    r0: np.ndarray,
    v0: np.ndarray,
    box: float,
    mass: Union[float, np.ndarray],
    dt: float,
    cutoff: float,
    config: VerifyTaskRunConfig | None = None,
    **legacy_kwargs: Any,
) -> VerifyResult:
    """Public verify-task entry point with compact config object."""
    if config is not None and legacy_kwargs:
        keys = ", ".join(sorted(legacy_kwargs.keys()))
        raise TypeError(
            f"run_verify_task received both config and legacy keyword options ({keys}); "
            "use one style"
        )
    cfg = config if config is not None else VerifyTaskRunConfig.from_legacy_kwargs(legacy_kwargs)
    return _run_verify_task_legacy(
        potential=potential,
        r0=r0,
        v0=v0,
        box=box,
        mass=mass,
        dt=dt,
        cutoff=cutoff,
        atom_types=cfg.atom_types,
        cell_size=cfg.cell_size,
        zones_total=cfg.zones_total,
        zone_cells_w=cfg.zone_cells_w,
        zone_cells_s=cfg.zone_cells_s,
        zone_cells_pattern=cfg.zone_cells_pattern,
        traversal=cfg.traversal,
        buffer_k=cfg.buffer_k,
        skin_from_buffer=cfg.skin_from_buffer,
        use_verlet=cfg.use_verlet,
        verlet_k_steps=cfg.verlet_k_steps,
        steps=cfg.steps,
        observer_every=cfg.observer_every,
        tol_dr=cfg.tol_dr,
        tol_dv=cfg.tol_dv,
        tol_dE=cfg.tol_dE,
        tol_dT=cfg.tol_dT,
        tol_dP=cfg.tol_dP,
        decomposition=cfg.decomposition,
        zones_nx=cfg.zones_nx,
        zones_ny=cfg.zones_ny,
        zones_nz=cfg.zones_nz,
        sync_mode=cfg.sync_mode,
        device=cfg.device,
        strict_min_zone_width=cfg.strict_min_zone_width,
        ensemble_kind=cfg.ensemble_kind,
        thermostat=cfg.thermostat,
        barostat=cfg.barostat,
        chaos_mode=cfg.chaos_mode,
        chaos_seed=cfg.chaos_seed,
        chaos_delay_prob=cfg.chaos_delay_prob,
        case_name=cfg.case_name,
    )
