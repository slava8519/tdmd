from __future__ import annotations

import argparse
import os
from typing import Callable

import numpy as np

from .backend import resolve_backend
from .config import load_config
from .io import (
    export_lammps_data,
    export_lammps_in,
    load_task,
    task_to_arrays,
    validate_task_for_run,
)
from .output import OutputSpec, make_output_bundle
from .potentials import make_potential
from .serial import run_serial
from .state import init_positions, init_velocities
from .td_full_mpi import run_td_full_mpi_1d
from .td_local import run_td_local


def _resolve_every(req: int, fallback: int) -> int:
    if req and req > 0:
        return int(req)
    return int(fallback) if fallback and fallback > 0 else 1


def _parse_traj_channels(raw: str) -> tuple[str, ...]:
    txt = str(raw or "").strip()
    if not txt:
        return ()
    out: list[str] = []
    for token in txt.split(","):
        t = str(token).strip().lower()
        if not t or t == "basic":
            continue
        if t == "all":
            for x in ("unwrapped", "image", "force"):
                if x not in out:
                    out.append(x)
            continue
        if t not in ("unwrapped", "image", "force"):
            raise SystemExit(
                "invalid --traj-channels value "
                f"{token!r}; allowed: unwrapped,image,force,all"
            )
        if t not in out:
            out.append(t)
    return tuple(out)


def _make_output_observer(
    *,
    output,
    traj_every: int,
    metrics_every: int,
    box0: float,
) -> tuple[Callable, int]:
    def obs(step, r, v, box_cur=None):
        box_now = float(box0 if box_cur is None else box_cur)
        if output.traj is not None and traj_every > 0 and (step % traj_every == 0):
            output.traj.write(step, r, v, box_value=(box_now, box_now, box_now))
        if output.metrics is not None and metrics_every > 0 and (step % metrics_every == 0):
            output.metrics.write(step, r, v, buffer_value=0.0, box_value=box_now)

    obs_every = 1 if (output.traj is not None or output.metrics is not None) else 0
    return obs, obs_every


def _cmd_golden_gen(args) -> None:
    cfg = load_config(args.config)
    pot = make_potential(cfg.potential.kind, cfg.potential.params)
    from .verify_golden import generate_golden

    out = generate_golden(
        cfg=cfg,
        potential=pot,
        out_path=args.out,
        steps=int(args.steps),
        every=int(args.every),
    )
    print(f"[golden-gen] wrote {out}")
    raise SystemExit(0)


def _cmd_golden_check(args) -> None:
    cfg = load_config(args.config)
    pot = make_potential(cfg.potential.kind, cfg.potential.params)
    from .verify_golden import check_against_golden

    res = check_against_golden(
        cfg=cfg,
        potential=pot,
        golden_path=args.golden,
        steps=int(args.steps),
        every=int(args.every),
        tol_dE=float(args.tol_dE),
        tol_dT=float(args.tol_dT),
        tol_dP=float(args.tol_dP),
    )
    ok = True
    for r in res:
        print(
            f"[golden-check {r.case}] ok={r.ok} max|dE|={r.max_dE:.3e} "
            f"max|dT|={r.max_dT:.3e} max|dP|={r.max_dP:.3e}"
        )
        ok = ok and r.ok
    raise SystemExit(0 if ok else 2)


def _cmd_verifylab(args) -> None:
    cfg = load_config(args.config)
    pot = make_potential(cfg.potential.kind, cfg.potential.params)
    from .verify_lab import summarize, sweep_verify2, write_csv
    from .verify_plots import plot_metrics_csv

    def parse_bool(s: str) -> bool:
        return s.strip().lower() in ("1", "true", "yes", "y", "on")

    zones_total_list = args.zones_total if args.zones_total else [cfg.td.zones_total]
    use_verlet_list = [parse_bool(x) for x in args.use_verlet] if args.use_verlet else [cfg.td.use_verlet]
    verlet_k_steps_list = args.verlet_k_steps if args.verlet_k_steps else [cfg.td.verlet_k_steps]
    chaos_mode_list = [parse_bool(x) for x in args.chaos] if args.chaos else [False]
    chaos_delay_prob_list = args.chaos_delay_prob if args.chaos_delay_prob else [0.0]
    rows = sweep_verify2(
        cfg,
        pot,
        steps=max(1, int(args.steps)),
        every=max(1, int(args.every)),
        cases=args.case if args.case else None,
        zones_total_list=zones_total_list,
        use_verlet_list=use_verlet_list,
        verlet_k_steps_list=verlet_k_steps_list,
        chaos_mode_list=chaos_mode_list,
        chaos_delay_prob_list=chaos_delay_prob_list,
        chaos_seed=int(args.chaos_seed),
    )
    write_csv(rows, args.csv)
    summ = summarize(rows)
    print(
        f"[verifylab] wrote {args.csv}  total={summ['total']} ok={summ['ok']} fail={summ['fail']}"
    )
    for c, s in summ["by_case"].items():
        print(
            f"  - {c}: ok {s['ok']}/{s['total']}  "
            f"worst dE={s['worst']['max_dE']:.3e} dP={s['worst']['max_dP']:.3e}"
        )
    if args.plots:
        plot_metrics_csv(args.csv, args.plots)
        print(f"[verifylab] plots -> {args.plots}")
    raise SystemExit(0 if summ["fail"] == 0 else 2)


def _cmd_verify2(args) -> None:
    cfg = load_config(args.config)
    pot = make_potential(cfg.potential.kind, cfg.potential.params)
    from .testcases import default_cases
    from .verify_v2 import run_verify_v2

    cases = default_cases()
    if args.case:
        wanted = set(args.case)
        cases = [c for c in cases if c.name in wanted]
        if not cases:
            raise SystemExit(
                f"Unknown cases: {args.case}. Known: {[c.name for c in default_cases()]}"
            )
    res = run_verify_v2(
        potential=pot,
        mass=cfg.system.mass,
        dt=cfg.run.dt,
        cutoff=cfg.run.cutoff,
        cell_size=cfg.td.cell_size,
        zones_total=cfg.td.zones_total,
        zone_cells_w=cfg.td.zone_cells_w,
        zone_cells_s=cfg.td.zone_cells_s,
        zone_cells_pattern=cfg.td.zone_cells_pattern,
        traversal=cfg.td.traversal,
        buffer_k=cfg.td.buffer_k,
        skin_from_buffer=cfg.td.skin_from_buffer,
        use_verlet=cfg.td.use_verlet,
        verlet_k_steps=cfg.td.verlet_k_steps,
        steps=max(1, int(args.steps)),
        observer_every=max(1, int(args.every)),
        tol_dr=float(args.tol_dr),
        tol_dv=float(args.tol_dv),
        tol_dE=float(args.tol_dE),
        tol_dT=float(args.tol_dT),
        tol_dP=float(args.tol_dP),
        ensemble_kind=str(cfg.ensemble.kind),
        thermostat=cfg.ensemble.thermostat,
        barostat=cfg.ensemble.barostat,
        cases=cases,
    )
    ok = True
    for r in res:
        print(
            f"[verify2 {r.case}] ok={r.ok}  max|dr|={r.max_dr:.3e} max|dv|={r.max_dv:.3e}  "
            f"max|dE|={r.max_dE:.3e} max|dT|={r.max_dT:.3e} max|dP|={r.max_dP:.3e}",
            flush=True,
        )
        ok = ok and r.ok
    raise SystemExit(0 if ok else 2)


def _cmd_run(args) -> None:
    cfg = load_config(args.config) if args.config else None

    task = load_task(args.task) if args.task else None
    if task is None and cfg is None:
        raise SystemExit("config required for run (or provide --task)")

    requested_device = str(args.device).strip().lower() if args.device else (
        str(getattr(cfg.run, "device", "auto")).strip().lower() if cfg is not None else "auto"
    )
    backend = resolve_backend(requested_device)
    if backend.device == "cuda":
        print("[backend] device=cuda (GPU path; CPU reference semantics preserved)", flush=True)
    else:
        reason = f" ({backend.reason})" if backend.reason else ""
        print(f"[backend] device=cpu{reason}", flush=True)

    if task is not None and args.export_lammps:
        outdir = args.export_lammps
        os.makedirs(outdir, exist_ok=True)
        data_path = os.path.join(outdir, "data.lammps")
        in_path = os.path.join(outdir, "in.lammps")
        export_lammps_data(task, data_path)
        export_lammps_in(task, in_path, data_filename=os.path.basename(data_path))
        print(f"[export-lammps] data -> {data_path}")
        print(f"[export-lammps] in   -> {in_path}")
        if args.export_only:
            return

    if task is not None:
        arr = task_to_arrays(task)
        allowed_kinds = ("lj", "morse", "table", "eam/alloy")
        allowed_ensembles = ("nve", "nvt", "npt")
        masses = validate_task_for_run(
            task,
            require_uniform_mass=False,
            allowed_potential_kinds=allowed_kinds,
            allowed_ensemble_kinds=allowed_ensembles,
        )
        r0 = arr.r.copy()
        v0 = arr.v.copy()
        box = float(task.box.x)
        dt = float(task.dt)
        cutoff = float(task.cutoff)
        n_steps = int(task.steps)
        thermo_every = int(cfg.run.thermo_every) if cfg is not None else 0
        pot = make_potential(task.potential.kind, task.potential.params)
        atom_ids = arr.atom_ids
        atom_types = arr.atom_types
        pbc = task.box.pbc
        mass = masses
        ensemble_kind = str(task.ensemble.kind)
        thermostat = task.ensemble.thermostat
        barostat = task.ensemble.barostat
    else:
        pot = make_potential(cfg.potential.kind, cfg.potential.params)
        r0 = init_positions(cfg.system.n_atoms, cfg.system.box, cfg.system.seed)
        v0 = init_velocities(cfg.system.n_atoms, cfg.system.temperature, cfg.system.mass, cfg.system.seed)
        mass = cfg.system.mass
        box = cfg.system.box
        dt = cfg.run.dt
        cutoff = cfg.run.cutoff
        n_steps = cfg.run.n_steps
        thermo_every = cfg.run.thermo_every
        atom_ids = np.arange(cfg.system.n_atoms, dtype=np.int32) + 1
        atom_types = np.ones(cfg.system.n_atoms, dtype=np.int32)
        pbc = (True, True, True)
        ensemble_kind = str(cfg.ensemble.kind)
        thermostat = cfg.ensemble.thermostat
        barostat = cfg.ensemble.barostat

    traj_channels = _parse_traj_channels(str(args.traj_channels))
    output_manifest = bool(not args.no_output_manifest)

    traj_every = _resolve_every(int(args.traj_every), int(thermo_every)) if args.traj else 0
    metrics_every = _resolve_every(int(args.metrics_every), int(thermo_every)) if args.metrics else 0

    output_spec = None
    if args.traj or args.metrics:
        output_spec = OutputSpec(
            traj_path=(args.traj if args.traj else None),
            traj_every=traj_every,
            metrics_path=(args.metrics if args.metrics else None),
            metrics_every=metrics_every,
            atom_ids=atom_ids,
            atom_types=atom_types,
            box=(box, box, box),
            pbc=pbc,
            mass=mass,
            cutoff=cutoff,
            potential=pot,
            traj_channels=traj_channels,
            traj_compression=str(args.traj_compression),
            write_output_manifest=output_manifest,
        )

    if args.mode == "serial":
        output = make_output_bundle(output_spec)
        obs, obs_every = _make_output_observer(
            output=output, traj_every=traj_every, metrics_every=metrics_every, box0=float(box)
        )
        run_serial(
            r0,
            v0,
            mass,
            box,
            pot,
            dt,
            cutoff,
            n_steps,
            thermo_every=thermo_every,
            observer=obs if obs_every else None,
            observer_every=obs_every,
            atom_types=atom_types,
            ensemble_kind=ensemble_kind,
            thermostat=thermostat,
            barostat=barostat,
            device=backend.device,
        )
        output.close()
        return

    if args.mode == "td_local":
        if cfg is None:
            raise SystemExit("td_local mode requires config for TD settings")
        output = make_output_bundle(output_spec)
        obs, obs_every = _make_output_observer(
            output=output, traj_every=traj_every, metrics_every=metrics_every, box0=float(box)
        )
        run_td_local(
            r0,
            v0,
            mass,
            box,
            pot,
            dt=dt,
            cutoff=cutoff,
            n_steps=n_steps,
            observer=obs if obs_every else None,
            observer_every=obs_every,
            atom_types=atom_types,
            cell_size=cfg.td.cell_size,
            zones_total=cfg.td.zones_total,
            zone_cells_w=cfg.td.zone_cells_w,
            zone_cells_s=cfg.td.zone_cells_s,
            zone_cells_pattern=cfg.td.zone_cells_pattern,
            traversal=cfg.td.traversal,
            buffer_k=cfg.td.buffer_k,
            skin_from_buffer=cfg.td.skin_from_buffer,
            use_verlet=cfg.td.use_verlet,
            verlet_k_steps=cfg.td.verlet_k_steps,
            decomposition=str(getattr(cfg.td, "decomposition", "1d")),
            zones_nx=int(getattr(cfg.td, "zones_nx", 1)),
            zones_ny=int(getattr(cfg.td, "zones_ny", 1)),
            zones_nz=int(getattr(cfg.td, "zones_nz", 1)),
            strict_min_zone_width=cfg.td.strict_min_zone_width,
            ensemble_kind=ensemble_kind,
            thermostat=thermostat,
            barostat=barostat,
            device=backend.device,
        )
        output.close()
        return

    if cfg is None:
        raise SystemExit("TD mode requires config for TD settings (use --mode serial or provide config)")

    run_td_full_mpi_1d(
        r=r0,
        v=v0,
        mass=mass,
        box=box,
        potential=pot,
        dt=dt,
        cutoff=cutoff,
        n_steps=n_steps,
        thermo_every=thermo_every,
        cell_size=cfg.td.cell_size,
        zones_total=cfg.td.zones_total,
        zone_cells_w=cfg.td.zone_cells_w,
        zone_cells_s=cfg.td.zone_cells_s,
        zone_cells_pattern=cfg.td.zone_cells_pattern,
        traversal=cfg.td.traversal,
        fast_sync=cfg.td.fast_sync,
        strict_fast_sync=cfg.td.strict_fast_sync,
        startup_mode=cfg.td.startup_mode,
        warmup_steps=cfg.td.warmup_steps,
        warmup_compute=cfg.td.warmup_compute,
        buffer_k=cfg.td.buffer_k,
        use_verlet=cfg.td.use_verlet,
        verlet_k_steps=cfg.td.verlet_k_steps,
        skin_from_buffer=cfg.td.skin_from_buffer,
        formal_core=cfg.td.formal_core,
        batch_size=cfg.td.batch_size,
        overlap_mode=cfg.td.overlap_mode,
        debug_invariants=cfg.td.debug_invariants,
        strict_min_zone_width=cfg.td.strict_min_zone_width,
        enable_step_id=cfg.td.enable_step_id,
        max_step_lag=cfg.td.max_step_lag,
        table_max_age=cfg.td.table_max_age,
        max_pending_delta_atoms=cfg.td.max_pending_delta_atoms,
        require_local_deps=cfg.td.require_local_deps,
        require_table_deps=cfg.td.require_table_deps,
        require_owner_deps=cfg.td.require_owner_deps,
        require_owner_ver=cfg.td.require_owner_ver,
        enable_req_holder=cfg.td.enable_req_holder,
        holder_gossip=cfg.td.holder_gossip,
        deps_provider_mode=cfg.td.deps_provider_mode,
        zones_nx=cfg.td.zones_nx,
        zones_ny=cfg.td.zones_ny,
        zones_nz=cfg.td.zones_nz,
        owner_buffer=cfg.td.owner_buffer,
        cuda_aware_mpi=cfg.td.cuda_aware_mpi,
        comm_overlap_isend=cfg.td.comm_overlap_isend,
        atom_types=atom_types,
        ensemble_kind=ensemble_kind,
        thermostat=thermostat,
        barostat=barostat,
        device=backend.device,
        output_spec=output_spec,
        trace_enabled=bool(args.trace_td),
        trace_path=str(args.trace_td_out),
    )
    return


def _cmd_verify(args) -> None:
    cfg = load_config(args.config)
    pot = make_potential(cfg.potential.kind, cfg.potential.params)
    r0 = init_positions(cfg.system.n_atoms, cfg.system.box, cfg.system.seed)
    v0 = init_velocities(cfg.system.n_atoms, cfg.system.temperature, cfg.system.mass, cfg.system.seed)
    steps = max(1, int(args.steps))
    rA = r0.copy()
    vA = v0.copy()
    rB = r0.copy()
    vB = v0.copy()

    run_serial(
        rA,
        vA,
        cfg.system.mass,
        cfg.system.box,
        pot,
        cfg.run.dt,
        cfg.run.cutoff,
        steps,
        thermo_every=0,
        atom_types=None,
        ensemble_kind=str(cfg.ensemble.kind),
        thermostat=cfg.ensemble.thermostat,
        barostat=cfg.ensemble.barostat,
    )

    run_td_local(
        rB,
        vB,
        cfg.system.mass,
        cfg.system.box,
        pot,
        dt=cfg.run.dt,
        cutoff=cfg.run.cutoff,
        n_steps=steps,
        cell_size=cfg.td.cell_size,
        zones_total=cfg.td.zones_total,
        zone_cells_w=cfg.td.zone_cells_w,
        zone_cells_s=cfg.td.zone_cells_s,
        zone_cells_pattern=cfg.td.zone_cells_pattern,
        traversal=cfg.td.traversal,
        buffer_k=cfg.td.buffer_k,
        skin_from_buffer=cfg.td.skin_from_buffer,
        use_verlet=cfg.td.use_verlet,
        verlet_k_steps=cfg.td.verlet_k_steps,
        decomposition=str(getattr(cfg.td, "decomposition", "1d")),
        zones_nx=int(getattr(cfg.td, "zones_nx", 1)),
        zones_ny=int(getattr(cfg.td, "zones_ny", 1)),
        zones_nz=int(getattr(cfg.td, "zones_nz", 1)),
        strict_min_zone_width=cfg.td.strict_min_zone_width,
        ensemble_kind=str(cfg.ensemble.kind),
        thermostat=cfg.ensemble.thermostat,
        barostat=cfg.ensemble.barostat,
    )

    dr = np.sqrt(((rA - rB) ** 2).sum(axis=1))
    dv = np.sqrt(((vA - vB) ** 2).sum(axis=1))
    print(
        f"[verify] steps={steps} max|dr|={float(dr.max()):.6e}  max|dv|={float(dv.max()):.6e}",
        flush=True,
    )
    return


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tdmd")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run")
    pr.add_argument("config", nargs="?", help="YAML config (TD settings)")
    pr.add_argument("--task", default="", help="Task YAML (explicit atoms/box)")
    pr.add_argument("--mode", choices=["td_full_mpi", "td_local", "serial"], default="td_full_mpi")
    pr.add_argument("--export-lammps", default="", help="Output directory for LAMMPS data+in files")
    pr.add_argument("--export-only", action="store_true", help="Export LAMMPS files and exit")
    pr.add_argument("--traj", default="", help="LAMMPS trajectory output path (.lammpstrj)")
    pr.add_argument("--traj-every", type=int, default=0, help="Trajectory output period (steps)")
    pr.add_argument(
        "--traj-channels",
        default="",
        help="Comma-separated trajectory channels: unwrapped,image,force (or all)",
    )
    pr.add_argument(
        "--traj-compression",
        choices=["none", "gz"],
        default="none",
        help="Trajectory output compression",
    )
    pr.add_argument("--metrics", default="", help="Metrics CSV output path")
    pr.add_argument("--metrics-every", type=int, default=0, help="Metrics output period (steps)")
    pr.add_argument(
        "--no-output-manifest",
        action="store_true",
        help="Disable trajectory/metrics schema sidecar manifests",
    )
    pr.add_argument("--trace-td", action="store_true", help="Enable TD trace logging")
    pr.add_argument("--trace-td-out", default="td_trace.csv", help="TD trace output path")
    pr.add_argument("--device", choices=["auto", "cpu", "cuda"], default="", help="Compute device override")
    pr.set_defaults(func=_cmd_run)

    pv = sub.add_parser("verify")
    pv.add_argument("config")
    pv.add_argument("--steps", type=int, default=50)
    pv.set_defaults(func=_cmd_verify)

    pg = sub.add_parser("golden-gen")
    pg.add_argument("config")
    pg.add_argument("--out", default="tests/golden/default_golden.json")
    pg.add_argument("--steps", type=int, default=200)
    pg.add_argument("--every", type=int, default=10)
    pg.set_defaults(func=_cmd_golden_gen)

    pc = sub.add_parser("golden-check")
    pc.add_argument("config")
    pc.add_argument("--golden", default="tests/golden/default_golden.json")
    pc.add_argument("--steps", type=int, default=200)
    pc.add_argument("--every", type=int, default=10)
    pc.add_argument("--tol_dE", type=float, default=1e-4)
    pc.add_argument("--tol_dT", type=float, default=1e-4)
    pc.add_argument("--tol_dP", type=float, default=1e-3)
    pc.set_defaults(func=_cmd_golden_check)

    pl = sub.add_parser("verifylab")
    pl.add_argument("config")
    pl.add_argument("--steps", type=int, default=200)
    pl.add_argument("--every", type=int, default=10)
    pl.add_argument("--case", action="append", default=[])
    pl.add_argument("--zones_total", action="append", type=int, default=[])
    pl.add_argument("--use_verlet", action="append", default=[])  # true/false
    pl.add_argument("--verlet_k_steps", action="append", type=int, default=[])
    pl.add_argument("--csv", default="verifylab_results.csv")
    pl.add_argument("--plots", default="")  # dir for pngs
    pl.add_argument("--chaos", action="append", default=[])  # true/false
    pl.add_argument("--chaos_delay_prob", action="append", type=float, default=[])
    pl.add_argument("--chaos_seed", type=int, default=12345)
    pl.set_defaults(func=_cmd_verifylab)

    pv2 = sub.add_parser("verify2")
    pv2.add_argument("config")
    pv2.add_argument("--steps", type=int, default=200)
    pv2.add_argument("--every", type=int, default=10)
    pv2.add_argument("--case", action="append", default=[])
    pv2.add_argument("--tol_dr", type=float, default=1e-6)
    pv2.add_argument("--tol_dv", type=float, default=1e-6)
    pv2.add_argument("--tol_dE", type=float, default=1e-5)
    pv2.add_argument("--tol_dT", type=float, default=1e-5)
    pv2.add_argument("--tol_dP", type=float, default=1e-4)
    pv2.set_defaults(func=_cmd_verify2)

    return p


def main() -> None:
    p = _build_parser()
    args = p.parse_args()
    func = getattr(args, "func", None)
    if func is None:
        raise SystemExit(f"unsupported cmd: {getattr(args, 'cmd', None)}")
    func(args)


if __name__ == "__main__":
    main()
