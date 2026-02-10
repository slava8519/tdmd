from __future__ import annotations

import argparse
from typing import Callable


def build_parser(
    *,
    cmd_run: Callable,
    cmd_verify: Callable,
    cmd_golden_gen: Callable,
    cmd_golden_check: Callable,
    cmd_verifylab: Callable,
    cmd_verify2: Callable,
) -> argparse.ArgumentParser:
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
    pr.set_defaults(func=cmd_run)

    pv = sub.add_parser("verify")
    pv.add_argument("config")
    pv.add_argument("--steps", type=int, default=50)
    pv.set_defaults(func=cmd_verify)

    pg = sub.add_parser("golden-gen")
    pg.add_argument("config")
    pg.add_argument("--out", default="tests/golden/default_golden.json")
    pg.add_argument("--steps", type=int, default=200)
    pg.add_argument("--every", type=int, default=10)
    pg.set_defaults(func=cmd_golden_gen)

    pc = sub.add_parser("golden-check")
    pc.add_argument("config")
    pc.add_argument("--golden", default="tests/golden/default_golden.json")
    pc.add_argument("--steps", type=int, default=200)
    pc.add_argument("--every", type=int, default=10)
    pc.add_argument("--tol_dE", type=float, default=1e-4)
    pc.add_argument("--tol_dT", type=float, default=1e-4)
    pc.add_argument("--tol_dP", type=float, default=1e-3)
    pc.set_defaults(func=cmd_golden_check)

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
    pl.set_defaults(func=cmd_verifylab)

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
    pv2.set_defaults(func=cmd_verify2)

    return p

