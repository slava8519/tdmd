from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import csv
import os
from datetime import datetime

from .verify_v2 import run_verify_v2, run_verify_config, run_verify_task, VerifyResult
from .io import load_task, task_to_arrays, validate_task_for_run
from .potentials import make_potential
from .testcases import default_cases

DEFAULT_THRESHOLDS = dict(dr=1e-6, dv=1e-6, dE=1e-5, dT=1e-5, dP=1e-4)

_TOL_ALIAS = {
    "tol_dr": "dr",
    "tol_dv": "dv",
    "tol_dE": "dE",
    "tol_dT": "dT",
    "tol_dP": "dP",
}

def _normalize_tol(tol: dict[str, float] | None) -> dict[str, float]:
    """Accept both dr/dv/dE... and tol_dr/tol_dv... keys."""
    if tol is None:
        merged = dict(DEFAULT_THRESHOLDS)
    else:
        merged = dict(tol)
    for src, dst in _TOL_ALIAS.items():
        if src in merged and dst not in merged:
            merged[dst] = merged[src]
    for k, v in DEFAULT_THRESHOLDS.items():
        merged.setdefault(k, v)
    return merged

@dataclass
class LabRow:
    case: str
    zones_total: int
    use_verlet: bool
    verlet_k_steps: int
    chaos_mode: bool
    chaos_delay_prob: float
    steps: int
    every: int
    ok: bool
    max_dr: float
    max_dv: float
    max_dE: float
    max_dT: float
    max_dP: float
    final_dr: float
    final_dv: float
    final_dE: float
    final_dT: float
    final_dP: float
    rms_dr: float
    rms_dv: float
    rms_dE: float
    rms_dT: float
    rms_dP: float
    details: dict[str, Any]

def sweep_verify2(cfg, potential,
                 *,
                 steps: int,
                 every: int,
                 zones_total_list: list[int] | None = None,
                 use_verlet_list: list[bool] | None = None,
                 verlet_k_steps_list: list[int] | None = None,
                 chaos_mode_list: list[bool] | None = None,
                 chaos_delay_prob_list: list[float] | None = None,
                 chaos_seed: int | None = None,
                 tol: dict[str, float] | None = None,
                 cases_mode: str = "cfg",
                 cases=None,
                 sync_mode: bool = False,
                 device: str = "cpu",
                 strict_min_zone_width: bool = False) -> list[LabRow]:
    """Experimental sweep.

    cases_mode:
      - 'cfg': compare on random system defined by YAML (Config.system)
      - 'testcases': compare on synthetic cases from tdmd.testcases (gas/crystal)
    """
    if zones_total_list is None: zones_total_list = [int(cfg.td.zones_total)]
    if use_verlet_list is None: use_verlet_list = [bool(cfg.td.use_verlet)]
    if verlet_k_steps_list is None: verlet_k_steps_list = [int(cfg.td.verlet_k_steps)]
    if chaos_mode_list is None: chaos_mode_list = [bool(cfg.td.chaos_mode)]
    if chaos_delay_prob_list is None: chaos_delay_prob_list = [float(cfg.td.chaos_delay_prob)]

    tol = _normalize_tol(tol)
    chaos_seed_val = int(cfg.td.chaos_seed) if chaos_seed is None else int(chaos_seed)

    if cases is None and str(cases_mode) == "testcases":
        cases = default_cases()

    rows: list[LabRow] = []
    ens_cfg = getattr(cfg, "ensemble", None)
    ensemble_kind = str(getattr(ens_cfg, "kind", "nve"))
    ensemble_thermostat = getattr(ens_cfg, "thermostat", None)
    ensemble_barostat = getattr(ens_cfg, "barostat", None)
    for zt in zones_total_list:
        for uv in use_verlet_list:
            for vk in verlet_k_steps_list:
                for cm in chaos_mode_list:
                    for cd in chaos_delay_prob_list:
                        if str(cases_mode) == "cfg":
                            res = run_verify_config(
                                potential=potential,
                                n_atoms=int(cfg.system.n_atoms),
                                box=float(cfg.system.box),
                                temperature=float(cfg.system.temperature),
                                seed=int(cfg.system.seed),
                                mass=float(cfg.system.mass),
                                dt=float(cfg.run.dt),
                                cutoff=float(cfg.run.cutoff),
                                cell_size=float(cfg.td.cell_size),
                                zones_total=int(zt),
                                zone_cells_w=int(cfg.td.zone_cells_w),
                                zone_cells_s=int(cfg.td.zone_cells_s),
                                zone_cells_pattern=cfg.td.zone_cells_pattern,
                                traversal=str(cfg.td.traversal),
                                buffer_k=float(cfg.td.buffer_k),
                                skin_from_buffer=bool(cfg.td.skin_from_buffer),
                                use_verlet=bool(uv),
                                verlet_k_steps=int(vk),
                                steps=int(steps),
                                observer_every=int(every),
                                tol_dr=float(tol["dr"]), tol_dv=float(tol["dv"]),
                                tol_dE=float(tol["dE"]), tol_dT=float(tol["dT"]), tol_dP=float(tol["dP"]),
                                decomposition=str(getattr(cfg.td, "decomposition", "1d")),
                                zones_nx=int(getattr(cfg.td, "zones_nx", 1)),
                                zones_ny=int(getattr(cfg.td, "zones_ny", 1)),
                                zones_nz=int(getattr(cfg.td, "zones_nz", 1)),
                                chaos_mode=bool(cm),
                                chaos_seed=chaos_seed_val,
                                chaos_delay_prob=float(cd),
                                sync_mode=bool(sync_mode),
                                ensemble_kind=ensemble_kind,
                                thermostat=ensemble_thermostat,
                                barostat=ensemble_barostat,
                                device=str(device),
                                strict_min_zone_width=bool(strict_min_zone_width),
                            )
                            res_list = [res]
                        else:
                            res_list = run_verify_v2(
                                potential=potential,
                                mass=float(cfg.system.mass),
                                dt=float(cfg.run.dt),
                                cutoff=float(cfg.run.cutoff),
                                cell_size=float(cfg.td.cell_size),
                                zones_total=int(zt),
                                zone_cells_w=int(cfg.td.zone_cells_w),
                                zone_cells_s=int(cfg.td.zone_cells_s),
                                zone_cells_pattern=cfg.td.zone_cells_pattern,
                                traversal=str(cfg.td.traversal),
                                buffer_k=float(cfg.td.buffer_k),
                                skin_from_buffer=bool(cfg.td.skin_from_buffer),
                                use_verlet=bool(uv),
                                verlet_k_steps=int(vk),
                                steps=int(steps),
                                observer_every=int(every),
                                tol_dr=float(tol["dr"]), tol_dv=float(tol["dv"]),
                                tol_dE=float(tol["dE"]), tol_dT=float(tol["dT"]), tol_dP=float(tol["dP"]),
                                cases=cases,
                                decomposition=str(getattr(cfg.td, "decomposition", "1d")),
                                zones_nx=int(getattr(cfg.td, "zones_nx", 1)),
                                zones_ny=int(getattr(cfg.td, "zones_ny", 1)),
                                zones_nz=int(getattr(cfg.td, "zones_nz", 1)),
                                chaos_mode=bool(cm),
                                chaos_seed=chaos_seed_val,
                                chaos_delay_prob=float(cd),
                                sync_mode=bool(sync_mode),
                                ensemble_kind=ensemble_kind,
                                thermostat=ensemble_thermostat,
                                barostat=ensemble_barostat,
                                device=str(device),
                                strict_min_zone_width=bool(strict_min_zone_width),
                            )

                        for r in res_list:
                            metrics = r.details.get("metrics", {})
                            rows.append(LabRow(
                                case=str(r.case),
                                zones_total=int(zt),
                                use_verlet=bool(uv),
                                verlet_k_steps=int(vk),
                                chaos_mode=bool(cm),
                                chaos_delay_prob=float(cd),
                                steps=int(r.steps),
                                every=int(every),
                                ok=bool(r.ok),
                                max_dr=float(r.max_dr),
                                max_dv=float(r.max_dv),
                                max_dE=float(r.max_dE),
                                max_dT=float(r.max_dT),
                                max_dP=float(r.max_dP),
                                final_dr=float(metrics.get("final_dr", 0.0)),
                                final_dv=float(metrics.get("final_dv", 0.0)),
                                final_dE=float(metrics.get("final_dE", 0.0)),
                                final_dT=float(metrics.get("final_dT", 0.0)),
                                final_dP=float(metrics.get("final_dP", 0.0)),
                                rms_dr=float(metrics.get("rms_dr", 0.0)),
                                rms_dv=float(metrics.get("rms_dv", 0.0)),
                                rms_dE=float(metrics.get("rms_dE", 0.0)),
                                rms_dT=float(metrics.get("rms_dT", 0.0)),
                                rms_dP=float(metrics.get("rms_dP", 0.0)),
                                details=dict(r.details),
                            ))
    return rows

def sweep_verify_task(task_path: str, td_cfg, *,
                 steps: int,
                 every: int,
                 zones_total_list: list[int] | None = None,
                 use_verlet_list: list[bool] | None = None,
                 verlet_k_steps_list: list[int] | None = None,
                 chaos_mode_list: list[bool] | None = None,
                 chaos_delay_prob_list: list[float] | None = None,
                 chaos_seed: int | None = None,
                 tol: dict[str, float] | None = None,
                 sync_mode: bool = False,
                 device: str = "cpu",
                 strict_min_zone_width: bool = False) -> list[LabRow]:
    """Verify serial vs TD-local using an explicit task file for initial state."""
    task = load_task(task_path)
    masses = validate_task_for_run(
        task,
        allowed_potential_kinds=("lj", "morse", "table", "eam/alloy"),
        allowed_ensemble_kinds=("nve", "nvt", "npt"),
    )
    arr = task_to_arrays(task)
    pot = make_potential(task.potential.kind, task.potential.params)
    box = float(task.box.x)
    dt = float(task.dt)
    cutoff = float(task.cutoff)
    steps = int(steps) if steps is not None else int(task.steps)

    if zones_total_list is None: zones_total_list = [int(td_cfg.zones_total)]
    if use_verlet_list is None: use_verlet_list = [bool(td_cfg.use_verlet)]
    if verlet_k_steps_list is None: verlet_k_steps_list = [int(td_cfg.verlet_k_steps)]
    if chaos_mode_list is None: chaos_mode_list = [bool(td_cfg.chaos_mode)]
    if chaos_delay_prob_list is None: chaos_delay_prob_list = [float(td_cfg.chaos_delay_prob)]

    tol = _normalize_tol(tol)
    chaos_seed_val = int(td_cfg.chaos_seed) if chaos_seed is None else int(chaos_seed)
    ensemble_kind = str(task.ensemble.kind)
    ensemble_thermostat = task.ensemble.thermostat
    ensemble_barostat = task.ensemble.barostat

    rows: list[LabRow] = []
    for zt in zones_total_list:
        for uv in use_verlet_list:
            for vk in verlet_k_steps_list:
                for cm in chaos_mode_list:
                    for cd in chaos_delay_prob_list:
                        r = run_verify_task(
                            potential=pot,
                            r0=arr.r, v0=arr.v, box=box,
                            mass=masses, dt=dt, cutoff=cutoff, atom_types=arr.atom_types,
                            cell_size=float(td_cfg.cell_size),
                            zones_total=int(zt),
                            zone_cells_w=int(td_cfg.zone_cells_w),
                            zone_cells_s=int(td_cfg.zone_cells_s),
                            zone_cells_pattern=td_cfg.zone_cells_pattern,
                            traversal=str(td_cfg.traversal),
                            buffer_k=float(td_cfg.buffer_k),
                            skin_from_buffer=bool(td_cfg.skin_from_buffer),
                            use_verlet=bool(uv),
                            verlet_k_steps=int(vk),
                            steps=int(steps),
                            observer_every=int(every),
                            tol_dr=float(tol["dr"]), tol_dv=float(tol["dv"]),
                            tol_dE=float(tol["dE"]), tol_dT=float(tol["dT"]), tol_dP=float(tol["dP"]),
                            decomposition=str(getattr(td_cfg, "decomposition", "1d")),
                            zones_nx=int(getattr(td_cfg, "zones_nx", 1)),
                            zones_ny=int(getattr(td_cfg, "zones_ny", 1)),
                            zones_nz=int(getattr(td_cfg, "zones_nz", 1)),
                            chaos_mode=bool(cm),
                            chaos_seed=chaos_seed_val,
                            chaos_delay_prob=float(cd),
                            sync_mode=bool(sync_mode),
                            ensemble_kind=ensemble_kind,
                            thermostat=ensemble_thermostat,
                            barostat=ensemble_barostat,
                            device=str(device),
                            strict_min_zone_width=bool(strict_min_zone_width),
                            case_name="interop_task",
                        )

                        metrics = r.details.get("metrics", {})
                        rows.append(LabRow(
                            case=str(r.case),
                            zones_total=int(zt),
                            use_verlet=bool(uv),
                            verlet_k_steps=int(vk),
                            chaos_mode=bool(cm),
                            chaos_delay_prob=float(cd),
                            steps=int(r.steps),
                            every=int(every),
                            ok=bool(r.ok),
                            max_dr=float(r.max_dr),
                            max_dv=float(r.max_dv),
                            max_dE=float(r.max_dE),
                            max_dT=float(r.max_dT),
                            max_dP=float(r.max_dP),
                            final_dr=float(metrics.get("final_dr", 0.0)),
                            final_dv=float(metrics.get("final_dv", 0.0)),
                            final_dE=float(metrics.get("final_dE", 0.0)),
                            final_dT=float(metrics.get("final_dT", 0.0)),
                            final_dP=float(metrics.get("final_dP", 0.0)),
                            rms_dr=float(metrics.get("rms_dr", 0.0)),
                            rms_dv=float(metrics.get("rms_dv", 0.0)),
                            rms_dE=float(metrics.get("rms_dE", 0.0)),
                            rms_dT=float(metrics.get("rms_dT", 0.0)),
                            rms_dP=float(metrics.get("rms_dP", 0.0)),
                            details=dict(r.details),
                        ))
    return rows

def write_csv(rows: list[LabRow], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case","zones_total","use_verlet","verlet_k_steps","chaos_mode","chaos_delay_prob","steps","every","ok",
                    "max_dr","max_dv","max_dE","max_dT","max_dP",
                    "final_dr","final_dv","final_dE","final_dT","final_dP",
                    "rms_dr","rms_dv","rms_dE","rms_dT","rms_dP"])
        for r in rows:
            w.writerow([r.case,r.zones_total,int(r.use_verlet),r.verlet_k_steps,int(r.chaos_mode),r.chaos_delay_prob,r.steps,r.every,int(r.ok),
                        r.max_dr,r.max_dv,r.max_dE,r.max_dT,r.max_dP,
                        r.final_dr,r.final_dv,r.final_dE,r.final_dT,r.final_dP,
                        r.rms_dr,r.rms_dv,r.rms_dE,r.rms_dT,r.rms_dP])

def _worst_metrics(rows: list[LabRow]) -> dict[str, float]:
    worst = {}
    for nm in ["max_dr", "max_dv", "max_dE", "max_dT", "max_dP"]:
        worst[nm] = max(getattr(r, nm) for r in rows)
    return worst

def summarize(rows: list[LabRow]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "total": 0, "ok": 0, "fail": 0, "ok_all": False, "worst": {}, "by_case": {}}
    total = len(rows)
    ok_count = sum(1 for r in rows if r.ok)
    fail_count = total - ok_count
    by_case: dict[str, Any] = {}
    for r in rows:
        by_case.setdefault(r.case, []).append(r)
    by_case_summ = {}
    for case, case_rows in by_case.items():
        case_total = len(case_rows)
        case_ok = sum(1 for r in case_rows if r.ok)
        by_case_summ[case] = {
            "total": case_total,
            "ok": case_ok,
            "fail": case_total - case_ok,
            "worst": _worst_metrics(case_rows),
        }
    return {
        "n": total,
        "total": total,
        "ok": ok_count,
        "fail": fail_count,
        "ok_all": bool(ok_count == total),
        "worst": _worst_metrics(rows),
        "by_case": by_case_summ,
    }

def summarize_markdown(rows: list[LabRow]) -> str:
    s = summarize(rows)
    lines = []
    lines.append(f"- n_rows: `{s.get('n')}`")
    ok_all = s.get("ok_all")
    if ok_all is None:
        ok_all = bool(s.get("ok"))
    lines.append(f"- ok: `{ok_all}`")
    w = s.get("worst", {})
    if w:
        lines.append("## Worst metrics")
        for k,v in w.items():
            lines.append(f"- {k}: `{v}`")
    bad = [r for r in rows if not r.ok]
    if bad:
        lines.append("\n## First failing rows")
        for r in bad[:5]:
            lines.append(f"- case={r.case} zt={r.zones_total} verlet={r.use_verlet} chaos={r.chaos_mode} "
                         f"dr={r.max_dr} dv={r.max_dv} dE={r.max_dE} dT={r.max_dT} dP={r.max_dP}")
    return "\n".join(lines)
