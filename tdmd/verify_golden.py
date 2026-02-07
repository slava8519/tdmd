from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .golden import GoldenSeries, load as load_golden, save as save_golden
from .observables import compute_observables
from .testcases import default_cases, make_case_state

@dataclass
class GoldenCheckResult:
    case: str
    ok: bool
    max_dE: float
    max_dT: float
    max_dP: float
    details: dict

def generate_golden(*, cfg, potential, out_path: str, steps: int, every: int, cases=None):
    from .serial import run_serial
    if cases is None:
        cases = default_cases()

    series=[]
    for case in cases:
        r0, v0, box = make_case_state(case, cfg.system.mass)
        obs=[]
        def cb(step, r, v):
            o=compute_observables(r, v, cfg.system.mass, box, potential, cfg.run.cutoff)
            obs.append((int(step), o))
        r=r0.copy(); v=v0.copy()
        run_serial(r, v, cfg.system.mass, box, potential, cfg.run.dt, cfg.run.cutoff, int(steps),
                   thermo_every=0, observer=cb, observer_every=int(every))
        obs_map=dict(obs)
        st=sorted(obs_map.keys())
        series.append(GoldenSeries(
            case=case.name,
            steps=st,
            E=[float(obs_map[s]["E"]) for s in st],
            T=[float(obs_map[s]["T"]) for s in st],
            P=[float(obs_map[s]["P"]) for s in st],
        ))
    save_golden(out_path, series)
    return out_path

def check_against_golden(*, cfg, potential, golden_path: str, steps: int, every: int,
                         tol_dE: float = 1e-4, tol_dT: float = 1e-4, tol_dP: float = 1e-3,
                         td_kwargs: dict | None = None):
    from .td_local import run_td_local
    if td_kwargs is None:
        td_kwargs = {}
    golden = {s.case: s for s in load_golden(golden_path)}
    cases = default_cases()
    results=[]
    for case in cases:
        if case.name not in golden:
            continue
        g=golden[case.name]
        r0, v0, box = make_case_state(case, cfg.system.mass)
        obs=[]
        def cb(step, r, v):
            o=compute_observables(r, v, cfg.system.mass, box, potential, cfg.run.cutoff)
            obs.append((int(step), o))
        r=r0.copy(); v=v0.copy()
        run_td_local(r, v, cfg.system.mass, box, potential, cfg.run.dt, cfg.run.cutoff, int(steps),
                     observer=cb, observer_every=int(every),
                     cell_size=cfg.td.cell_size, zones_total=td_kwargs.get("zones_total", cfg.td.zones_total),
                     zone_cells_w=cfg.td.zone_cells_w, zone_cells_s=cfg.td.zone_cells_s,
                     zone_cells_pattern=cfg.td.zone_cells_pattern, traversal=cfg.td.traversal,
                     buffer_k=cfg.td.buffer_k, skin_from_buffer=cfg.td.skin_from_buffer,
                     use_verlet=td_kwargs.get("use_verlet", cfg.td.use_verlet),
                     verlet_k_steps=td_kwargs.get("verlet_k_steps", cfg.td.verlet_k_steps))
        obs_map=dict(obs)
        # compare on common steps
        common=[s for s in g.steps if s in obs_map]
        dE=[]; dT=[]; dP=[]
        for i,s in enumerate(common):
            dE.append(abs(obs_map[s]["E"] - g.E[i]))
            dT.append(abs(obs_map[s]["T"] - g.T[i]))
            dP.append(abs(obs_map[s]["P"] - g.P[i]))
        max_dE=float(max(dE)) if dE else 0.0
        max_dT=float(max(dT)) if dT else 0.0
        max_dP=float(max(dP)) if dP else 0.0
        ok=(max_dE<=tol_dE and max_dT<=tol_dT and max_dP<=tol_dP)
        results.append(GoldenCheckResult(
            case=case.name, ok=bool(ok), max_dE=max_dE, max_dT=max_dT, max_dP=max_dP,
            details={"common_steps": common}
        ))
    return results
