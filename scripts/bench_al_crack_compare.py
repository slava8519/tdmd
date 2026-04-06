#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from generate_al_crack_task import build_al_crack_state, write_al_crack_task_yaml
from tdmd.backend import resolve_backend
from tdmd.forces_gpu import reset_device_state_cache
from tdmd.io import TelemetryWriter, load_task, task_to_arrays, validate_task_for_run
from tdmd.potentials import make_potential
from tdmd.td_local import run_td_local
from tdmd.zones import ZoneLayout1DCells


def _device_case_tag(device: str) -> str:
    return "gpu" if str(device).strip().lower() == "cuda" else "cpu"


def _factorizations(total: int) -> list[tuple[int, int, int]]:
    out: set[tuple[int, int, int]] = set()
    for nx in range(1, int(total) + 1):
        if int(total) % nx != 0:
            continue
        rem = int(total) // nx
        for ny in range(1, rem + 1):
            if rem % ny != 0:
                continue
            nz = rem // ny
            out.add(tuple(sorted((int(nx), int(ny), int(nz)), reverse=True)))
    return sorted(out)


def _best_space_layout(total: int, box: float, cutoff: float) -> tuple[int, int, int]:
    best: tuple[float, float, tuple[int, int, int]] | None = None
    for nx, ny, nz in _factorizations(total):
        widths = [float(box) / float(nx), float(box) / float(ny), float(box) / float(nz)]
        if min(widths) < float(cutoff):
            continue
        score = (min(widths) / max(widths), min(widths))
        if best is None or score > (best[0], best[1]):
            best = (score[0], score[1], (int(nx), int(ny), int(nz)))
    if best is None:
        raise ValueError(f"no valid 3D layout for zones_total={total} and cutoff={cutoff}")
    return best[2]


def _td_preflight(box: float, cutoff: float, cell_size: float, zones_total: int) -> None:
    layout = ZoneLayout1DCells(
        box=float(box),
        cell_size=float(cell_size),
        zones_total=int(zones_total),
        zone_cells_w=1,
        zone_cells_s=1,
        min_zone_width=float(cutoff),
        strict_min_width=True,
    )
    layout.build()


def _max_valid_td_zones(box: float, cutoff: float, cell_size: float) -> int:
    n_cells_total = max(1, int(round(float(box) / float(cell_size))))
    dz_cell = float(box) / float(n_cells_total)
    min_cells = max(1, int(math.ceil(float(cutoff) / dz_cell)))
    return max(1, n_cells_total // min_cells)


def _fmt_time(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}s"


def _fmt_rate(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _fmt_speedup(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}x"


def _load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_last_telemetry(telemetry_path: Path) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    summary = _load_json(Path(f"{telemetry_path}.summary.json"))
    if summary is not None:
        last = summary.get("last_record", {})
        if isinstance(last, dict):
            return summary, dict(last)
    if not telemetry_path.exists():
        return summary, None
    try:
        lines = [line for line in telemetry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return summary, None
        payload = json.loads(lines[-1])
        return summary, payload if isinstance(payload, dict) else None
    except Exception:
        return summary, None


def _speedup(space_row: dict[str, object] | None, time_row: dict[str, object] | None) -> float | None:
    if not space_row or not time_row:
        return None
    t_space = float(space_row.get("elapsed_sec", 0.0) or 0.0)
    t_time = float(time_row.get("elapsed_sec", 0.0) or 0.0)
    if t_space <= 0.0 or t_time <= 0.0:
        return None
    return float(t_space / t_time)


def _run_requested_td_preflight(*, box: float, cutoff: float, cell_size: float, zones_total: int) -> dict[str, object]:
    out = {
        "case": f"time_z{int(zones_total)}",
        "ok": False,
        "zones_total": int(zones_total),
        "elapsed_sec": None,
        "steps_per_sec": None,
    }
    try:
        _td_preflight(float(box), float(cutoff), float(cell_size), int(zones_total))
        out["preflight_ok"] = True
    except Exception as exc:
        out["preflight_ok"] = False
        out["error"] = str(exc)
    return out


def _worker_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--worker-out", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--requested-device", required=True)
    ap.add_argument("--require-effective-cuda", action="store_true")
    ap.add_argument("--decomposition", required=True, choices=["1d", "3d"])
    ap.add_argument("--zones-total", type=int, required=True)
    ap.add_argument("--zones-nx", type=int, required=True)
    ap.add_argument("--zones-ny", type=int, required=True)
    ap.add_argument("--zones-nz", type=int, required=True)
    ap.add_argument("--cell-size", type=float, required=True)
    ap.add_argument("--telemetry-path", required=True)
    ap.add_argument("--telemetry-every", type=int, required=True)
    ap.add_argument("--telemetry-heartbeat-sec", type=float, required=True)
    ap.add_argument("--telemetry-stdout", action="store_true")
    return ap


def _run_worker(args) -> int:
    task = load_task(str(args.task))
    arr = task_to_arrays(task)
    masses = validate_task_for_run(task, allowed_potential_kinds=("eam/alloy",))
    potential = make_potential(task.potential.kind, task.potential.params)
    backend = resolve_backend(str(args.requested_device))
    effective_device = str(getattr(backend, "device", "cpu"))
    fallback_from_cuda = bool(str(args.requested_device) == "cuda" and effective_device != "cuda")
    telemetry = TelemetryWriter(
        str(args.telemetry_path),
        total_steps=int(task.steps),
        mass=masses,
        atom_count=int(arr.r.shape[0]),
        device=effective_device,
        mode=f"td_local/{str(args.decomposition)}",
        emit_stdout=bool(args.telemetry_stdout),
        heartbeat_every_sec=float(args.telemetry_heartbeat_sec),
        metadata={
            "case": str(args.label),
            "requested_device": str(args.requested_device),
            "zones_total": int(args.zones_total),
            "space_layout": f"{int(args.zones_nx)}x{int(args.zones_ny)}x{int(args.zones_nz)}",
        },
    )

    def _write_result_and_exit(*, ok: bool, error: str, elapsed_sec: float | None, exit_code: int) -> int:
        last = telemetry.close(completed=bool(ok)) or {}
        payload = {
            "case": str(args.label),
            "ok": bool(ok),
            "requested_device": str(args.requested_device),
            "effective_device": effective_device,
            "fallback_from_cuda": int(fallback_from_cuda),
            "decomposition": str(args.decomposition),
            "zones_total": int(args.zones_total),
            "space_layout": f"{int(args.zones_nx)}x{int(args.zones_ny)}x{int(args.zones_nz)}",
            "elapsed_sec": None if elapsed_sec is None else float(elapsed_sec),
            "steps_per_sec": (
                float(task.steps) / float(elapsed_sec)
                if elapsed_sec is not None and float(elapsed_sec) > 0.0
                else None
            ),
            "error": str(error),
            "telemetry_path": str(args.telemetry_path),
            "telemetry_summary_path": f"{str(args.telemetry_path)}.summary.json",
            "last_step_observed": int(last.get("step", 0)) if last else None,
            "last_wall_sec": float(last.get("wall_sec", 0.0) or 0.0) if last else None,
            "last_rss_mb": float(last.get("rss_mb", 0.0) or 0.0) if last else None,
            "last_gpu_device_used_mb": (
                float(last.get("gpu_device_used_mb", 0.0) or 0.0) if last else None
            ),
        }
        Path(args.worker_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return int(exit_code)

    def _signal_handler(signum, _frame):
        code = 124 if int(signum) == int(signal.SIGTERM) else 130
        raise SystemExit(code)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    if bool(args.require_effective_cuda) and effective_device != "cuda":
        return _write_result_and_exit(
            ok=False,
            error="requested device=cuda but backend resolved to cpu",
            elapsed_sec=None,
            exit_code=2,
        )

    try:
        if str(args.decomposition) == "1d":
            _td_preflight(float(task.box.x), float(task.cutoff), float(args.cell_size), int(args.zones_total))
        r = np.asarray(arr.r, dtype=np.float64).copy()
        v = np.asarray(arr.v, dtype=np.float64).copy()
        reset_device_state_cache()
        started = time.perf_counter()
        run_td_local(
            r=r,
            v=v,
            mass=masses,
            box=float(task.box.x),
            potential=potential,
            dt=float(task.dt),
            cutoff=float(task.cutoff),
            n_steps=int(task.steps),
            observer=telemetry,
            observer_every=max(1, int(args.telemetry_every)),
            atom_types=arr.atom_types,
            cell_size=float(args.cell_size),
            zones_total=int(args.zones_total),
            zone_cells_w=1,
            zone_cells_s=1,
            traversal="forward",
            buffer_k=1.2,
            skin_from_buffer=True,
            use_verlet=True,
            verlet_k_steps=max(10, int(task.steps)),
            decomposition=str(args.decomposition),
            sync_mode=False,
            zones_nx=int(args.zones_nx),
            zones_ny=int(args.zones_ny),
            zones_nz=int(args.zones_nz),
            strict_min_zone_width=True,
            ensemble_kind="nve",
            device=str(args.requested_device),
        )
        elapsed_sec = float(time.perf_counter() - started)
        return _write_result_and_exit(ok=True, error="", elapsed_sec=elapsed_sec, exit_code=0)
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 1
        msg = "terminated" if code == 124 else "interrupted"
        return _write_result_and_exit(ok=False, error=msg, elapsed_sec=None, exit_code=code)
    except Exception as exc:
        return _write_result_and_exit(ok=False, error=str(exc), elapsed_sec=None, exit_code=1)


def _run_case_subprocess(
    *,
    task_path: str,
    label: str,
    requested_device: str,
    require_effective_cuda: bool,
    decomposition: str,
    zones_total: int,
    zones_nx: int,
    zones_ny: int,
    zones_nz: int,
    cell_size: float,
    timeout_sec: int,
    telemetry_dir: str,
    telemetry_every: int,
    telemetry_heartbeat_sec: float,
    telemetry_stdout: bool,
) -> dict[str, object]:
    base = Path(telemetry_dir)
    base.mkdir(parents=True, exist_ok=True)
    telemetry_path = base / f"{label}.telemetry.jsonl"
    worker_out = base / f"{label}.worker.json"
    if telemetry_path.exists():
        telemetry_path.unlink()
    if Path(f"{telemetry_path}.summary.json").exists():
        Path(f"{telemetry_path}.summary.json").unlink()
    if worker_out.exists():
        worker_out.unlink()

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-out",
        str(worker_out),
        "--task",
        str(task_path),
        "--label",
        str(label),
        "--requested-device",
        str(requested_device),
        "--decomposition",
        str(decomposition),
        "--zones-total",
        str(int(zones_total)),
        "--zones-nx",
        str(int(zones_nx)),
        "--zones-ny",
        str(int(zones_ny)),
        "--zones-nz",
        str(int(zones_nz)),
        "--cell-size",
        str(float(cell_size)),
        "--telemetry-path",
        str(telemetry_path),
        "--telemetry-every",
        str(max(1, int(telemetry_every))),
        "--telemetry-heartbeat-sec",
        str(float(telemetry_heartbeat_sec)),
    ]
    if bool(require_effective_cuda):
        cmd.append("--require-effective-cuda")
    if bool(telemetry_stdout):
        cmd.append("--telemetry-stdout")

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT_DIR),
        stdout=None if bool(telemetry_stdout) else subprocess.PIPE,
        stderr=None if bool(telemetry_stdout) else subprocess.PIPE,
        text=True,
    )
    timed_out = False
    stdout = ""
    stderr = ""
    try:
        comm_out, comm_err = proc.communicate(timeout=max(1, int(timeout_sec)))
        if comm_out is not None:
            stdout = str(comm_out)
        if comm_err is not None:
            stderr = str(comm_err)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.terminate()
        try:
            comm_out, comm_err = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            comm_out, comm_err = proc.communicate()
        if comm_out is not None:
            stdout = str(comm_out)
        if comm_err is not None:
            stderr = str(comm_err)
    rc = int(proc.returncode if proc.returncode is not None else 124)

    payload = _load_json(worker_out) or {}
    summary, last = _load_last_telemetry(telemetry_path)
    row = {
        "case": str(label),
        "ok": bool(payload.get("ok", False)),
        "requested_device": str(payload.get("requested_device", requested_device)),
        "effective_device": str(payload.get("effective_device", "")),
        "fallback_from_cuda": int(payload.get("fallback_from_cuda", 0) or 0),
        "decomposition": str(payload.get("decomposition", decomposition)),
        "zones_total": int(payload.get("zones_total", zones_total) or zones_total),
        "space_layout": str(payload.get("space_layout", f"{zones_nx}x{zones_ny}x{zones_nz}")),
        "elapsed_sec": payload.get("elapsed_sec"),
        "steps_per_sec": payload.get("steps_per_sec"),
        "error": str(payload.get("error", "")),
        "telemetry_path": str(telemetry_path),
        "telemetry_summary_path": f"{str(telemetry_path)}.summary.json",
        "last_step_observed": payload.get("last_step_observed"),
        "last_wall_sec": payload.get("last_wall_sec"),
        "last_rss_mb": payload.get("last_rss_mb"),
        "last_gpu_device_used_mb": payload.get("last_gpu_device_used_mb"),
        "worker_out": str(worker_out),
        "worker_returncode": rc,
        "worker_timed_out": bool(timed_out),
        "worker_stdout": str(stdout or ""),
        "worker_stderr": str(stderr or ""),
    }
    if last is not None:
        row["last_step_observed"] = int(last.get("step", row.get("last_step_observed") or 0))
        row["last_wall_sec"] = float(last.get("wall_sec", row.get("last_wall_sec") or 0.0) or 0.0)
        row["last_rss_mb"] = float(last.get("rss_mb", row.get("last_rss_mb") or 0.0) or 0.0)
        row["last_gpu_device_used_mb"] = float(
            last.get("gpu_device_used_mb", row.get("last_gpu_device_used_mb") or 0.0) or 0.0
        )
    row["telemetry_completed"] = bool(summary.get("completed", False)) if summary else False
    if timed_out and not row["error"]:
        row["error"] = f"benchmark case timed out after {int(timeout_sec)}s"
    if timed_out:
        row["ok"] = False
        row["worker_returncode"] = 124
    return row


def _build_markdown_report(
    *,
    task_path: str,
    generated_task: bool,
    n_atoms: int,
    box: float,
    cutoff: float,
    steps: int,
    requested_device: str,
    effective_device: str,
    requested_zones: int,
    max_valid_td_zones: int,
    requested_space_layout: tuple[int, int, int],
    requested_td_preflight: dict[str, object],
    compare_zones: int,
    rows: list[dict[str, object]],
    exact_speedup: float | None,
    common_speedup: float | None,
    budget_elapsed_sec: float,
) -> str:
    lines = [
        "# Al Crack Decomposition Benchmark",
        "",
        f"- task: `{task_path}`",
        f"- generated_task: `{int(bool(generated_task))}`",
        f"- n_atoms: `{int(n_atoms)}`",
        f"- steps: `{int(steps)}`",
        f"- box: `{float(box):.3f}`",
        f"- cutoff: `{float(cutoff):.3f}`",
        f"- requested_device: `{requested_device}`",
        f"- effective_device: `{effective_device}`",
        f"- requested_zones_total: `{int(requested_zones)}`",
        f"- max_valid_td_zones_total: `{int(max_valid_td_zones)}`",
        f"- requested_space_layout: `{requested_space_layout[0]}x{requested_space_layout[1]}x{requested_space_layout[2]}`",
        f"- strict_valid_common_zones_total: `{int(compare_zones)}`",
        "",
        "## Exact Request Preflight",
        "",
        f"- `time_z{int(requested_zones)}` preflight_ok=`{bool(requested_td_preflight.get('preflight_ok', False))}`",
    ]
    if requested_td_preflight.get("error"):
        lines.append(f"- `time_z{int(requested_zones)}` error=`{requested_td_preflight['error']}`")

    lines.extend(
        [
            "",
            "## Results",
            "",
            "| case | ok | decomposition | zones_total | layout | elapsed | steps_per_sec | last_step | timed_out | rss_mb | gpu_used_mb |",
            "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + str(row["case"])
            + " | "
            + str(int(bool(row.get("ok", False))))
            + " | "
            + str(row.get("decomposition", ""))
            + " | "
            + str(int(row.get("zones_total", 0) or 0))
            + " | "
            + str(row.get("space_layout", ""))
            + " | "
            + _fmt_time(row.get("elapsed_sec"))
            + " | "
            + _fmt_rate(row.get("steps_per_sec"))
            + " | "
            + ("n/a" if row.get("last_step_observed") is None else str(int(row.get("last_step_observed", 0))))
            + " | "
            + str(int(bool(row.get("worker_timed_out", False))))
            + " | "
            + ("n/a" if row.get("last_rss_mb") is None else f"{float(row.get('last_rss_mb')):.1f}")
            + " | "
            + (
                "n/a"
                if row.get("last_gpu_device_used_mb") is None
                else f"{float(row.get('last_gpu_device_used_mb')):.1f}"
            )
            + " |"
        )
        if row.get("error"):
            lines.append(f"- error `{row['case']}`: `{row['error']}`")
        if row.get("telemetry_summary_path"):
            lines.append(f"- telemetry `{row['case']}`: `{row['telemetry_summary_path']}`")

    lines.extend(
        [
            "",
            "## Comparison",
            "",
            f"- exact_request_td_speedup_vs_space: `{_fmt_speedup(exact_speedup)}`",
            f"- strict_valid_td_speedup_vs_space: `{_fmt_speedup(common_speedup)}` at common zones_total `{int(compare_zones)}`",
            f"- wall_time_total: `{float(budget_elapsed_sec):.3f}s`",
        ]
    )
    return "\n".join(lines) + "\n"


def _main_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Benchmark a pure-Al microcrack task under TD vs space decomposition")
    ap.add_argument("--task", default="")
    ap.add_argument("--task-out", default="")
    ap.add_argument("--out", default="results/al_crack_100k_compare_gpu.csv")
    ap.add_argument("--md", default="results/al_crack_100k_compare_gpu.md")
    ap.add_argument("--json", default="results/al_crack_100k_compare_gpu.summary.json")
    ap.add_argument("--target-atoms", type=int, default=100000)
    ap.add_argument("--box", type=float, default=122.0)
    ap.add_argument("--lattice-a", type=float, default=4.05)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--cutoff", type=float, default=6.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--requested-zones", type=int, default=1000)
    ap.add_argument("--cell-size", type=float, default=0.5)
    ap.add_argument(
        "--eam-file",
        default="examples/potentials/eam_alloy/Al_zhou.eam.alloy",
        help="Single-element Al EAM file used in the generated task",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--velocity-std", type=float, default=0.020)
    ap.add_argument("--timeout-sec", type=int, default=600)
    ap.add_argument("--requested-space-timeout-sec", type=int, default=180)
    ap.add_argument("--compare-space-timeout-sec", type=int, default=180)
    ap.add_argument("--compare-time-timeout-sec", type=int, default=180)
    ap.add_argument("--telemetry-dir", default="")
    ap.add_argument("--telemetry-every", type=int, default=1)
    ap.add_argument("--telemetry-heartbeat-sec", type=float, default=5.0)
    ap.add_argument("--telemetry-stdout", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--require-effective-cuda", action="store_true")
    return ap


def main() -> int:
    worker_parser = _worker_parser()
    if "--worker" in sys.argv[1:]:
        worker_args = worker_parser.parse_args()
        return _run_worker(worker_args)

    ap = _main_parser()
    args = ap.parse_args()

    out_csv = Path(args.out)
    out_md = Path(args.md)
    out_json = Path(args.json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    telemetry_dir = str(
        Path(args.telemetry_dir) if args.telemetry_dir else out_json.parent / f"{out_json.stem}_telemetry"
    )
    Path(telemetry_dir).mkdir(parents=True, exist_ok=True)

    if args.task:
        task_path = str(args.task)
        generated_task = False
        crack_lo = crack_hi = None
        available_after_crack = None
    else:
        task_out = (
            Path(args.task_out)
            if args.task_out
            else out_json.parent / f"{out_json.stem}.task.yaml"
        )
        positions, types, masses, velocities, crack_lo, crack_hi, available_after_crack = build_al_crack_state(
            target_atoms=int(args.target_atoms),
            box=float(args.box),
            lattice_a=float(args.lattice_a),
            seed=int(args.seed),
            velocity_std=float(args.velocity_std),
        )
        write_al_crack_task_yaml(
            out_path=task_out,
            box=float(args.box),
            dt=float(args.dt),
            steps=int(args.steps),
            cutoff=float(args.cutoff),
            positions=positions,
            types=types,
            masses=masses,
            velocities=velocities,
            eam_file=str(args.eam_file),
        )
        task_path = str(task_out)
        generated_task = True

    backend = resolve_backend(str(args.device))
    effective_device = str(getattr(backend, "device", "cpu"))
    if bool(args.require_effective_cuda) and effective_device != "cuda":
        raise SystemExit("hardware-strict GPU benchmark requires effective device=cuda")

    task = load_task(str(task_path))
    cutoff = float(task.cutoff)
    box = float(task.box.x)
    requested_zones = int(args.requested_zones)
    bench_start = time.perf_counter()
    device_tag = _device_case_tag(str(args.device))
    requested_space_layout = _best_space_layout(requested_zones, box, cutoff)
    requested_td = _run_requested_td_preflight(
        box=box,
        cutoff=cutoff,
        cell_size=float(args.cell_size),
        zones_total=requested_zones,
    )
    compare_zones = min(
        requested_zones,
        _max_valid_td_zones(box=box, cutoff=cutoff, cell_size=float(args.cell_size)),
    )
    compare_space_layout = _best_space_layout(compare_zones, box, cutoff)

    rows: list[dict[str, object]] = []
    remaining = int(args.timeout_sec)

    exact_space_row = _run_case_subprocess(
        task_path=str(task_path),
        label=f"space_{device_tag}_z{requested_zones}",
        requested_device=str(args.device),
        require_effective_cuda=bool(args.require_effective_cuda),
        decomposition="3d",
        zones_total=requested_zones,
        zones_nx=requested_space_layout[0],
        zones_ny=requested_space_layout[1],
        zones_nz=requested_space_layout[2],
        cell_size=float(args.cell_size),
        timeout_sec=min(max(1, remaining), int(args.requested_space_timeout_sec)),
        telemetry_dir=telemetry_dir,
        telemetry_every=int(args.telemetry_every),
        telemetry_heartbeat_sec=float(args.telemetry_heartbeat_sec),
        telemetry_stdout=bool(args.telemetry_stdout),
    )
    rows.append(exact_space_row)
    remaining = max(0, int(args.timeout_sec) - int(time.perf_counter() - bench_start))

    exact_time_row = None
    common_space_row = None
    common_time_row = None

    if bool(requested_td.get("preflight_ok", False)):
        exact_time_row = _run_case_subprocess(
            task_path=str(task_path),
            label=f"time_{device_tag}_z{requested_zones}",
            requested_device=str(args.device),
            require_effective_cuda=bool(args.require_effective_cuda),
            decomposition="1d",
            zones_total=requested_zones,
            zones_nx=1,
            zones_ny=1,
            zones_nz=1,
            cell_size=float(args.cell_size),
            timeout_sec=min(max(1, remaining), int(args.compare_time_timeout_sec)),
            telemetry_dir=telemetry_dir,
            telemetry_every=int(args.telemetry_every),
            telemetry_heartbeat_sec=float(args.telemetry_heartbeat_sec),
            telemetry_stdout=bool(args.telemetry_stdout),
        )
        rows.append(exact_time_row)
    else:
        remaining = max(0, int(args.timeout_sec) - int(time.perf_counter() - bench_start))
        if remaining > 0:
            common_space_row = _run_case_subprocess(
                task_path=str(task_path),
                label=f"space_{device_tag}_z{compare_zones}",
                requested_device=str(args.device),
                require_effective_cuda=bool(args.require_effective_cuda),
                decomposition="3d",
                zones_total=compare_zones,
                zones_nx=compare_space_layout[0],
                zones_ny=compare_space_layout[1],
                zones_nz=compare_space_layout[2],
                cell_size=float(args.cell_size),
                timeout_sec=min(max(1, remaining), int(args.compare_space_timeout_sec)),
                telemetry_dir=telemetry_dir,
                telemetry_every=int(args.telemetry_every),
                telemetry_heartbeat_sec=float(args.telemetry_heartbeat_sec),
                telemetry_stdout=bool(args.telemetry_stdout),
            )
            rows.append(common_space_row)
        remaining = max(0, int(args.timeout_sec) - int(time.perf_counter() - bench_start))
        if remaining > 0:
            common_time_row = _run_case_subprocess(
                task_path=str(task_path),
                label=f"time_{device_tag}_z{compare_zones}",
                requested_device=str(args.device),
                require_effective_cuda=bool(args.require_effective_cuda),
                decomposition="1d",
                zones_total=compare_zones,
                zones_nx=1,
                zones_ny=1,
                zones_nz=1,
                cell_size=float(args.cell_size),
                timeout_sec=min(max(1, remaining), int(args.compare_time_timeout_sec)),
                telemetry_dir=telemetry_dir,
                telemetry_every=int(args.telemetry_every),
                telemetry_heartbeat_sec=float(args.telemetry_heartbeat_sec),
                telemetry_stdout=bool(args.telemetry_stdout),
            )
            rows.append(common_time_row)

    rows_by_case = {str(row["case"]): row for row in rows}
    if exact_time_row is None and requested_td.get("preflight_ok", False):
        exact_time_row = rows_by_case.get(f"time_{device_tag}_z{requested_zones}")
    exact_speedup = _speedup(
        rows_by_case.get(f"space_{device_tag}_z{requested_zones}"),
        rows_by_case.get(f"time_{device_tag}_z{requested_zones}"),
    )
    common_speedup = _speedup(
        rows_by_case.get(f"space_{device_tag}_z{compare_zones}"),
        rows_by_case.get(f"time_{device_tag}_z{compare_zones}"),
    )
    if common_speedup is None and compare_zones == requested_zones:
        common_speedup = exact_speedup

    report_markdown = _build_markdown_report(
        task_path=str(task_path),
        generated_task=bool(generated_task),
        n_atoms=int(len(task.atoms)),
        box=box,
        cutoff=cutoff,
        steps=int(task.steps),
        requested_device=str(args.device),
        effective_device=effective_device,
        requested_zones=requested_zones,
        max_valid_td_zones=compare_zones,
        requested_space_layout=requested_space_layout,
        requested_td_preflight=requested_td,
        compare_zones=compare_zones,
        rows=rows,
        exact_speedup=exact_speedup,
        common_speedup=common_speedup,
        budget_elapsed_sec=float(time.perf_counter() - bench_start),
    )

    ok_all = all(bool(row.get("ok", False)) for row in rows)
    if bool(args.require_effective_cuda):
        ok_all = bool(ok_all and all(str(row.get("effective_device", "cpu")) == "cuda" for row in rows))

    summary = {
        "n": len(rows),
        "total": len(rows),
        "ok": int(sum(1 for row in rows if bool(row.get("ok", False)))),
        "fail": int(sum(1 for row in rows if not bool(row.get("ok", False)))),
        "ok_all": bool(ok_all),
        "generated_task": bool(generated_task),
        "task_path": str(task_path),
        "n_atoms": int(len(task.atoms)),
        "box": float(box),
        "cutoff": float(cutoff),
        "requested_device": str(args.device),
        "effective_device": effective_device,
        "requested_zones_total": requested_zones,
        "requested_space_layout": {
            "zones_total": requested_zones,
            "layout": {
                "nx": int(requested_space_layout[0]),
                "ny": int(requested_space_layout[1]),
                "nz": int(requested_space_layout[2]),
            },
        },
        "requested_td_preflight": requested_td,
        "max_valid_td_zones_total": int(compare_zones),
        "strict_valid_common_zones_total": int(compare_zones),
        "strict_valid_common_space_layout": {
            "nx": int(compare_space_layout[0]),
            "ny": int(compare_space_layout[1]),
            "nz": int(compare_space_layout[2]),
        },
        "cell_size": float(args.cell_size),
        "rows": rows,
        "by_case": {str(row["case"]): dict(row) for row in rows},
        "comparisons": {
            "exact_request_td_speedup_vs_space": exact_speedup,
            "strict_valid_td_speedup_vs_space": common_speedup,
        },
        "budget": {
            "timeout_sec": int(args.timeout_sec),
            "requested_space_timeout_sec": int(args.requested_space_timeout_sec),
            "compare_space_timeout_sec": int(args.compare_space_timeout_sec),
            "compare_time_timeout_sec": int(args.compare_time_timeout_sec),
            "elapsed_wall_sec": float(time.perf_counter() - bench_start),
        },
        "telemetry": {
            "dir": telemetry_dir,
            "every": int(args.telemetry_every),
            "heartbeat_sec": float(args.telemetry_heartbeat_sec),
            "stdout": bool(args.telemetry_stdout),
        },
        "effective_cuda_required": bool(args.require_effective_cuda),
        "report_markdown": report_markdown,
        "artifacts": {
            "task": str(task_path),
            "csv": str(out_csv),
            "md": str(out_md),
            "json": str(out_json),
            "telemetry_dir": telemetry_dir,
        },
    }
    if generated_task:
        summary["crack_geometry"] = {
            "lo": [float(x) for x in np.asarray(crack_lo, dtype=float)],
            "hi": [float(x) for x in np.asarray(crack_hi, dtype=float)],
            "available_after_crack": int(available_after_crack),
        }

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "case",
                "ok",
                "requested_device",
                "effective_device",
                "fallback_from_cuda",
                "decomposition",
                "zones_total",
                "space_layout",
                "elapsed_sec",
                "steps_per_sec",
                "last_step_observed",
                "worker_timed_out",
                "last_wall_sec",
                "last_rss_mb",
                "last_gpu_device_used_mb",
                "error",
                "telemetry_path",
                "telemetry_summary_path",
            ]
        )
        for row in rows:
            w.writerow(
                [
                    row["case"],
                    int(bool(row.get("ok", False))),
                    row["requested_device"],
                    row["effective_device"],
                    int(row.get("fallback_from_cuda", 0) or 0),
                    row["decomposition"],
                    int(row.get("zones_total", 0) or 0),
                    row["space_layout"],
                    "" if row.get("elapsed_sec") is None else f"{float(row['elapsed_sec']):.6f}",
                    "" if row.get("steps_per_sec") is None else f"{float(row['steps_per_sec']):.6f}",
                    "" if row.get("last_step_observed") is None else int(row["last_step_observed"]),
                    int(bool(row.get("worker_timed_out", False))),
                    "" if row.get("last_wall_sec") is None else f"{float(row['last_wall_sec']):.6f}",
                    "" if row.get("last_rss_mb") is None else f"{float(row['last_rss_mb']):.6f}",
                    (
                        ""
                        if row.get("last_gpu_device_used_mb") is None
                        else f"{float(row['last_gpu_device_used_mb']):.6f}"
                    ),
                    row.get("error", ""),
                    row.get("telemetry_path", ""),
                    row.get("telemetry_summary_path", ""),
                ]
            )

    out_md.write_text(report_markdown, encoding="utf-8")
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(report_markdown, end="")
    print(f"[al-crack-compare] wrote {out_csv}")
    print(f"[al-crack-compare] wrote {out_md}")
    print(f"[al-crack-compare] wrote {out_json}")
    if bool(args.strict) and not bool(summary["ok_all"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
