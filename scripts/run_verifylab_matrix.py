from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import warnings
from datetime import datetime

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tdmd.backend import resolve_backend
from tdmd.config import load_config
from tdmd.incident_bundle import write_incident_bundle
from tdmd.potentials import make_potential
from tdmd.verify_lab import (
    summarize,
    summarize_markdown,
    sweep_verify2,
    sweep_verify_task,
    write_csv,
)

DEFAULT_THRESHOLDS = dict(tol_dr=1e-5, tol_dv=1e-5, tol_dE=1e-4, tol_dT=1e-4, tol_dP=1e-3)
SMOKE_CI_THRESHOLDS = dict(tol_dr=1e-5, tol_dv=3e-5, tol_dE=2.5e-4, tol_dT=1e-4, tol_dP=1e-3)

PRESETS = {
    # CI-grade strict smoke: short and expected to pass with --strict.
    "smoke_ci": dict(
        steps=2,
        every=1,
        zones_total_list=[4],
        use_verlet_list=[False],
        verlet_k_steps_list=[10],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
        tol=SMOKE_CI_THRESHOLDS,
    ),
    # Diagnostic regression smoke: keeps legacy behavior/coverage and may include non-ok rows.
    "smoke_regression": dict(
        steps=80,
        every=10,
        zones_total_list=[4],
        use_verlet_list=[False],
        verlet_k_steps_list=[10],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
        tol=DEFAULT_THRESHOLDS,
    ),
    # Backward-compatible alias.
    "smoke": dict(
        steps=80,
        every=10,
        zones_total_list=[4],
        use_verlet_list=[False],
        verlet_k_steps_list=[10],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
        tol=DEFAULT_THRESHOLDS,
    ),
    "paper": dict(
        steps=400,
        every=20,
        zones_total_list=[4, 8],
        use_verlet_list=[False, True],
        verlet_k_steps_list=[10],
        chaos_mode_list=[False, True],
        chaos_delay_prob_list=[0.0, 0.02],
    ),
    "stress": dict(
        steps=800,
        every=20,
        zones_total_list=[8, 16],
        use_verlet_list=[True],
        verlet_k_steps_list=[10],
        chaos_mode_list=[True],
        chaos_delay_prob_list=[0.05],
    ),
    "async": dict(
        steps=300,
        every=10,
        zones_total_list=[4, 8],
        use_verlet_list=[False],
        verlet_k_steps_list=[10],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
    ),
    "longrun_envelope_ci": dict(
        steps=300,
        every=10,
        zones_total_list=[4, 8],
        use_verlet_list=[False],
        verlet_k_steps_list=[10],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
        # Keep base verify gate permissive; strict envelope gate is applied from baseline.
        tol=dict(tol_dr=100.0, tol_dv=2.0, tol_dE=500.0, tol_dT=1.0, tol_dP=1e-3),
        envelope_file="golden/longrun_envelope_v1.json",
    ),
    "sync": dict(
        steps=200,
        every=10,
        zones_total_list=[4, 8],
        use_verlet_list=[False],
        verlet_k_steps_list=[10],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
        sync_mode=True,
    ),
    "paper_testcases_light": dict(
        steps=120,
        every=20,
        zones_total_list=[4],
        use_verlet_list=[False],
        verlet_k_steps_list=[10],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
    ),
    "interop_smoke": dict(
        steps=40,
        every=10,
        zones_total_list=[1],
        use_verlet_list=[False],
        verlet_k_steps_list=[10],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
        tol=DEFAULT_THRESHOLDS,
        task_mode=True,
    ),
    "nvt_smoke": dict(
        steps=40,
        every=10,
        zones_total_list=[1],
        use_verlet_list=[False],
        verlet_k_steps_list=[10],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
        tol=DEFAULT_THRESHOLDS,
        task_mode=True,
        sync_mode=True,
        task_path="examples/interop/task_nvt.yaml",
    ),
    "npt_smoke": dict(
        steps=40,
        every=10,
        zones_total_list=[1],
        use_verlet_list=[False],
        verlet_k_steps_list=[10],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
        tol=DEFAULT_THRESHOLDS,
        task_mode=True,
        sync_mode=True,
        task_path="examples/interop/task_npt.yaml",
    ),
    "metal_smoke": dict(
        steps=4,
        every=1,
        zones_total_list=[1],
        use_verlet_list=[False],
        verlet_k_steps_list=[10],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
        tol=SMOKE_CI_THRESHOLDS,
        task_mode=True,
        sync_mode=True,
        task_path="examples/interop/task_eam_al.yaml",
    ),
    "interop_metal_smoke": dict(
        steps=4,
        every=1,
        zones_total_list=[1],
        use_verlet_list=[False],
        verlet_k_steps_list=[10],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
        tol=SMOKE_CI_THRESHOLDS,
        task_mode=True,
        sync_mode=True,
        task_path="examples/interop/task_eam_alloy.yaml",
    ),
    "gpu_smoke": dict(
        steps=2,
        every=1,
        zones_total_list=[4],
        use_verlet_list=[False],
        verlet_k_steps_list=[10],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
        tol=SMOKE_CI_THRESHOLDS,
        sync_mode=True,
        device="cuda",
    ),
    "gpu_smoke_hw": dict(
        steps=2,
        every=1,
        zones_total_list=[4],
        use_verlet_list=[False],
        verlet_k_steps_list=[10],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
        tol=SMOKE_CI_THRESHOLDS,
        sync_mode=True,
        device="cuda",
        require_effective_cuda=True,
    ),
    "gpu_interop_smoke": dict(
        steps=2,
        every=1,
        zones_total_list=[1],
        use_verlet_list=[False],
        verlet_k_steps_list=[10],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
        tol=SMOKE_CI_THRESHOLDS,
        task_mode=True,
        sync_mode=True,
        device="cuda",
        task_path="examples/interop/task.yaml",
    ),
    "gpu_interop_smoke_hw": dict(
        steps=2,
        every=1,
        zones_total_list=[1],
        use_verlet_list=[False],
        verlet_k_steps_list=[10],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
        tol=SMOKE_CI_THRESHOLDS,
        task_mode=True,
        sync_mode=True,
        device="cuda",
        require_effective_cuda=True,
        task_path="examples/interop/task.yaml",
    ),
    "gpu_metal_smoke": dict(
        steps=2,
        every=1,
        zones_total_list=[1],
        use_verlet_list=[False],
        verlet_k_steps_list=[10],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
        tol=SMOKE_CI_THRESHOLDS,
        task_mode=True,
        sync_mode=True,
        device="cuda",
        task_path="examples/interop/task_eam_alloy_uniform_mass.yaml",
    ),
    "gpu_metal_smoke_hw": dict(
        steps=2,
        every=1,
        zones_total_list=[1],
        use_verlet_list=[False],
        verlet_k_steps_list=[10],
        chaos_mode_list=[False],
        chaos_delay_prob_list=[0.0],
        tol=SMOKE_CI_THRESHOLDS,
        task_mode=True,
        sync_mode=True,
        device="cuda",
        require_effective_cuda=True,
        task_path="examples/interop/task_eam_alloy_uniform_mass.yaml",
    ),
    "gpu_perf_smoke": dict(
        gpu_perf_mode=True,
        device="cuda",
        require_effective_cuda=True,
        timeout=180,
        n_atoms=65536,
        delta_count=256,
        repeats=9,
        max_delta_over_full=0.65,
        max_transfer_over_kernel=4.0,
    ),
    "mpi_overlap_smoke": dict(
        mpi_overlap_mode=True,
        mpi_config="examples/td_1d_morse_static_rr_smoke4.yaml",
        mpi_ranks=[2, 4],
        overlap_list="0,1",
        strict_invariants=True,
        timeout=180,
    ),
    "mpi_overlap_cudaaware_smoke": dict(
        mpi_overlap_mode=True,
        mpi_config="examples/td_1d_morse_static_rr_smoke4.yaml",
        mpi_ranks=[2, 4],
        overlap_list="0,1",
        strict_invariants=True,
        cuda_aware=True,
        timeout=180,
    ),
    "cluster_scale_smoke": dict(
        cluster_scale_mode=True,
        cluster_profile="examples/cluster/cluster_profile_smoke.yaml",
        timeout=300,
    ),
    "cluster_stability_smoke": dict(
        cluster_stability_mode=True,
        cluster_profile="examples/cluster/cluster_profile_smoke.yaml",
        timeout=300,
    ),
    "mpi_transport_matrix_smoke": dict(
        mpi_transport_matrix_mode=True,
        cluster_profile="examples/cluster/cluster_profile_smoke.yaml",
        timeout=300,
    ),
    "viz_smoke": dict(
        viz_contract_mode=True,
        cluster_profile="examples/cluster/cluster_profile_smoke.yaml",
        timeout=240,
    ),
    "metal_property_smoke": dict(
        materials_property_mode=True,
        materials_fixture="examples/interop/materials_parity_suite_v2.json",
        case_prefix="eam_al_",
        timeout=300,
    ),
    "interop_metal_property_smoke": dict(
        materials_property_mode=True,
        materials_fixture="examples/interop/materials_parity_suite_v2.json",
        case_prefix="eam_alloy_",
        timeout=300,
    ),
}


def _backend_evidence(requested_device: str) -> dict[str, object]:
    req = str(requested_device).strip().lower()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        b = resolve_backend(req)
    warn_txt = [str(x.message) for x in w]
    fallback = bool(req == "cuda" and b.device != "cuda")
    return {
        "requested_device": req,
        "effective_device": str(b.device),
        "cuda_available": bool(b.cuda_available),
        "fallback_from_cuda": fallback,
        "reason": str(b.reason),
        "warnings": warn_txt,
        "ok_effective_cuda": bool(req != "cuda" or b.device == "cuda"),
    }


def _to_bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return bool(int(v))
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _row_key_obj(r) -> tuple[object, ...]:
    return (
        str(r.case),
        int(r.zones_total),
        bool(r.use_verlet),
        int(r.verlet_k_steps),
        bool(r.chaos_mode),
        float(r.chaos_delay_prob),
    )


def _row_key_dict(x: dict[str, object]) -> tuple[object, ...]:
    return (
        str(x.get("case", "")),
        int(x.get("zones_total", 0)),
        _to_bool(x.get("use_verlet", False)),
        int(x.get("verlet_k_steps", 0)),
        _to_bool(x.get("chaos_mode", False)),
        float(x.get("chaos_delay_prob", 0.0)),
    )


def apply_envelope_gate(rows, envelope_file: str) -> dict[str, object]:
    out = {
        "enabled": True,
        "file": str(envelope_file),
        "version": 0,
        "rows_checked": 0,
        "rows_failed": 0,
        "missing_baseline_rows": 0,
        "ok_all": True,
        "violations": [],
    }
    if not rows:
        return out
    with open(envelope_file, "r", encoding="utf-8") as f:
        spec = json.load(f)
    out["version"] = int(spec.get("envelope_version", 0))
    base_rows = list(spec.get("rows", []))
    base_map = {_row_key_dict(x): x for x in base_rows}
    used_keys = set()

    for r in rows:
        out["rows_checked"] = int(out["rows_checked"]) + 1
        key = _row_key_obj(r)
        base = base_map.get(key)
        details = dict(r.details or {})
        violations = list(details.get("violations", []))
        env_meta = dict(details.get("envelope", {}))
        env_meta["baseline_key"] = {
            "case": str(r.case),
            "zones_total": int(r.zones_total),
            "use_verlet": bool(r.use_verlet),
            "verlet_k_steps": int(r.verlet_k_steps),
            "chaos_mode": bool(r.chaos_mode),
            "chaos_delay_prob": float(r.chaos_delay_prob),
        }

        if base is None:
            violations.append("envelope_missing_row")
            out["missing_baseline_rows"] = int(out["missing_baseline_rows"]) + 1
        else:
            used_keys.add(key)
            limits = dict(base.get("max", {}))
            for nm, lim in limits.items():
                if not hasattr(r, nm):
                    violations.append(f"envelope_unknown_metric:{nm}")
                    continue
                val = float(getattr(r, nm))
                limf = float(lim)
                if val > limf + 1e-18:
                    violations.append(f"envelope_{nm}: value={val:.6e} limit={limf:.6e}")
                env_meta.setdefault("checks", {})[str(nm)] = {"value": val, "limit": limf}

        if violations:
            r.ok = False
            out["rows_failed"] = int(out["rows_failed"]) + 1
            for vv in violations:
                out["violations"].append(
                    {
                        "case": str(r.case),
                        "zones_total": int(r.zones_total),
                        "violation": str(vv),
                    }
                )
        details["violations"] = violations
        details["envelope"] = env_meta
        r.details = details

    out["unused_baseline_rows"] = int(max(0, len(base_rows) - len(used_keys)))
    out["ok_all"] = bool(int(out["rows_failed"]) == 0)
    return out


def _detect_mpirun() -> str:
    env_val = os.environ.get("MPIRUN", "").strip()
    if env_val:
        return env_val
    local_candidates = [
        os.path.join(ROOT_DIR, ".venv", "bin", "mpiexec.hydra"),
        os.path.join(ROOT_DIR, ".venv", "bin", "mpiexec"),
        os.path.join(ROOT_DIR, ".venv", "bin", "mpirun"),
    ]
    for cand in local_candidates:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return shutil.which("mpiexec.hydra") or shutil.which("mpiexec") or shutil.which("mpirun") or ""


def _read_overlap_csv(path: str) -> list[dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(str(row.get(key, default)))
    except Exception:
        return int(default)


def _as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(str(row.get(key, default)))
    except Exception:
        return float(default)


def _pareto_frontier(points, *, x_key, y_key):
    """Compute Pareto frontier for minimizing x and maximizing y.
    Returns a list of points (original dicts) on the frontier, sorted by x ascending.
    """
    pts = [p for p in points if p is not None]
    pts.sort(key=lambda p: (float(p.get(x_key, 0.0)), -float(p.get(y_key, 0.0))))
    frontier = []
    best_y = -float("inf")
    for p in pts:
        _x = float(p.get(x_key, 0.0))  # noqa: F841 – kept for symmetry with sort key
        y = float(p.get(y_key, 0.0))
        if y > best_y:
            frontier.append(p)
            best_y = y
    return frontier


def _best_under(points, *, x_key, y_key, x_max):
    cand = [p for p in points if float(p.get(x_key, 0.0)) <= float(x_max)]
    if not cand:
        return None
    return max(cand, key=lambda p: float(p.get(y_key, 0.0)))


def _run_mpi_overlap_sweep(*, preset: dict, out_dir: str) -> dict[str, object]:
    ranks = [int(x) for x in list(preset.get("mpi_ranks", [2, 4]))]
    overlap_list = str(preset.get("overlap_list", "0,1"))
    mpi_config = str(preset.get("mpi_config", "examples/td_1d_morse_static_rr_smoke4.yaml"))
    timeout = int(preset.get("timeout", 180))
    cuda_aware = bool(preset.get("cuda_aware", False))
    strict_invariants = bool(preset.get("strict_invariants", True))
    mpirun = str(preset.get("mpirun", "")).strip() or _detect_mpirun()

    rank_runs: list[dict[str, object]] = []
    for n in ranks:
        out_csv = os.path.join(out_dir, f"mpi_overlap_n{int(n)}.csv")
        out_md = os.path.join(out_dir, f"mpi_overlap_n{int(n)}.md")
        cmd = [
            sys.executable,
            "scripts/bench_mpi_overlap.py",
            "--config",
            mpi_config,
            "--n",
            str(int(n)),
            "--overlap-list",
            overlap_list,
            "--timeout",
            str(timeout),
            "--out",
            out_csv,
            "--md",
            out_md,
        ]
        if mpirun:
            cmd.extend(["--mpirun", str(mpirun)])
        if cuda_aware:
            cmd.append("--cuda-aware")
        if not strict_invariants:
            cmd.append("--no-strict-invariants")

        proc = subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True)
        bench_rows = _read_overlap_csv(out_csv)
        run_rcs = [_as_int(r, "returncode", default=1) for r in bench_rows]
        strict_ok = [_as_int(r, "strict_invariants_ok", default=0) for r in bench_rows]
        hG_vals = [_as_int(r, "hG_max", default=0) for r in bench_rows]
        hV_vals = [_as_int(r, "hV_max", default=0) for r in bench_rows]
        violW_vals = [_as_int(r, "violW_max", default=0) for r in bench_rows]
        lagV_vals = [_as_int(r, "lagV_max", default=0) for r in bench_rows]
        asyncS_vals = [_as_int(r, "async_send_msgs_max", default=0) for r in bench_rows]
        asyncB_vals = [_as_int(r, "async_send_bytes_max", default=0) for r in bench_rows]
        wfgC_vals = [_as_int(r, "wfgC_max", default=0) for r in bench_rows]
        wfgO_vals = [_as_int(r, "wfgO_max", default=0) for r in bench_rows]
        wfgS_vals = [_as_int(r, "wfgS_max", default=0) for r in bench_rows]
        wfgC_rate_vals = [float(r.get("wfgC_rate", 0.0) or 0.0) for r in bench_rows]
        wfgC_p100_vals = [float(r.get("wfgC_per_100_steps", 0.0) or 0.0) for r in bench_rows]
        diag_vals = [_as_int(r, "diag_samples", default=0) for r in bench_rows]
        speedup_overlap = [
            _as_float(r, "speedup_vs_blocking", default=0.0)
            for r in bench_rows
            if _as_int(r, "overlap", default=0) == 1
        ]

        run_ok = bool(
            proc.returncode == 0
            and len(bench_rows) > 0
            and all(int(x) == 0 for x in run_rcs)
            and (not strict_invariants or all(int(x) == 1 for x in strict_ok))
        )
        reasons: list[str] = []
        if not mpirun:
            reasons.append("mpirun_not_found")
        if proc.returncode != 0:
            reasons.append(f"bench_rc={int(proc.returncode)}")
        if not bench_rows:
            reasons.append("missing_bench_rows")
        if run_rcs and any(int(x) != 0 for x in run_rcs):
            reasons.append("inner_run_failed")
        if strict_invariants and strict_ok and any(int(x) != 1 for x in strict_ok):
            reasons.append("strict_invariants_failed")

        rank_runs.append(
            {
                "ranks": int(n),
                "ok": bool(run_ok),
                "bench_returncode": int(proc.returncode),
                "records": int(len(bench_rows)),
                "overlap_speedup": float(min(speedup_overlap) if speedup_overlap else 0.0),
                "hG_max": int(max(hG_vals) if hG_vals else 0),
                "hV_max": int(max(hV_vals) if hV_vals else 0),
                "violW_max": int(max(violW_vals) if violW_vals else 0),
                "lagV_max": int(max(lagV_vals) if lagV_vals else 0),
                "async_send_msgs_max": int(max(asyncS_vals) if asyncS_vals else 0),
                "async_send_bytes_max": int(max(asyncB_vals) if asyncB_vals else 0),
                "wfgC_max": int(max(wfgC_vals) if wfgC_vals else 0),
                "wfgO_max": int(max(wfgO_vals) if wfgO_vals else 0),
                "wfgS_max": int(max(wfgS_vals) if wfgS_vals else 0),
                "wfgC_rate": float(max(wfgC_rate_vals) if wfgC_rate_vals else 0.0),
                "wfgC_per_100_steps": float(max(wfgC_p100_vals) if wfgC_p100_vals else 0.0),
                "diag_samples_min": int(min(diag_vals) if diag_vals else 0),
                "reasons": reasons,
                "out_csv": out_csv,
                "out_md": out_md,
            }
        )

    total = int(len(rank_runs))
    ok_n = int(sum(1 for r in rank_runs if bool(r.get("ok", False))))
    fail_n = int(total - ok_n)
    summary: dict[str, object] = {
        "n": total,
        "total": total,
        "ok": ok_n,
        "fail": fail_n,
        "ok_all": bool(ok_n == total),
        "worst": {
            "max_hG": max((int(r["hG_max"]) for r in rank_runs), default=0),
            "max_hV": max((int(r["hV_max"]) for r in rank_runs), default=0),
            "max_violW": max((int(r["violW_max"]) for r in rank_runs), default=0),
            "max_lagV": max((int(r["lagV_max"]) for r in rank_runs), default=0),
            "max_async_send_msgs": max(
                (int(r.get("async_send_msgs_max", 0)) for r in rank_runs), default=0
            ),
            "max_async_send_bytes": max(
                (int(r.get("async_send_bytes_max", 0)) for r in rank_runs), default=0
            ),
            "min_overlap_speedup": min(
                (float(r["overlap_speedup"]) for r in rank_runs), default=0.0
            ),
        },
        "by_case": {},
        "mpi_overlap_runs": rank_runs,
    }
    by_case: dict[str, object] = {}
    for r in rank_runs:
        case = f"mpi_overlap_n{int(r['ranks'])}"
        by_case[case] = {
            "total": 1,
            "ok": int(bool(r["ok"])),
            "fail": int(not bool(r["ok"])),
            "worst": {
                "hG_max": int(r["hG_max"]),
                "hV_max": int(r["hV_max"]),
                "violW_max": int(r["violW_max"]),
                "lagV_max": int(r["lagV_max"]),
                "async_send_msgs_max": int(r.get("async_send_msgs_max", 0)),
                "async_send_bytes_max": int(r.get("async_send_bytes_max", 0)),
                "wfgC_max": int(r.get("wfgC_max", 0)),
                "wfgO_max": int(r.get("wfgO_max", 0)),
                "wfgS_max": int(r.get("wfgS_max", 0)),
                "wfgC_rate": float(r.get("wfgC_rate", 0.0) or 0.0),
                "wfgC_per_100_steps": float(r.get("wfgC_per_100_steps", 0.0) or 0.0),
            },
        }
    summary["by_case"] = by_case
    return summary


def _run_external_cluster_script(
    *, script_rel: str, preset: dict, out_dir: str
) -> dict[str, object]:
    profile = str(preset.get("cluster_profile", "examples/cluster/cluster_profile_smoke.yaml"))
    timeout = int(preset.get("timeout", 300))
    script_name = os.path.splitext(os.path.basename(script_rel))[0]
    out_csv = os.path.join(out_dir, f"{script_name}.csv")
    out_md = os.path.join(out_dir, f"{script_name}.md")
    out_json = os.path.join(out_dir, f"{script_name}.summary.json")
    cmd = [
        sys.executable,
        script_rel,
        "--profile",
        profile,
        "--out",
        out_csv,
        "--md",
        out_md,
        "--json",
        out_json,
        "--strict",
    ]
    timed_out = False
    proc_rc = 1
    proc_out = ""
    proc_err = ""
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT_DIR, capture_output=True, text=True, timeout=max(1, timeout)
        )
        proc_rc = int(proc.returncode)
        proc_out = str(proc.stdout or "")
        proc_err = str(proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        proc_rc = 124
        proc_out = str(getattr(exc, "stdout", "") or "")
        proc_err = str(getattr(exc, "stderr", "") or "")

    summary: dict[str, object] = {
        "n": 0,
        "total": 0,
        "ok": 0,
        "fail": 1,
        "ok_all": False,
        "worst": {},
        "by_case": {},
        "external_script": str(script_rel),
        "external_returncode": int(proc_rc),
        "external_timed_out": bool(timed_out),
        "external_stdout": proc_out,
        "external_stderr": proc_err,
        "external_profile": profile,
        "external_artifacts": {"csv": out_csv, "md": out_md, "json": out_json},
    }
    if os.path.exists(out_json):
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                summary.update(loaded)
        except Exception as exc:
            summary["parse_error"] = str(exc)
    summary["ok_all"] = bool(summary.get("ok_all", False) and int(proc_rc) == 0 and not timed_out)
    if bool(summary.get("ok_all", False)):
        summary["ok"] = int(summary.get("ok", summary.get("total", summary.get("n", 0))))
        summary["fail"] = int(summary.get("fail", 0))
    else:
        total = int(summary.get("total", summary.get("n", 1)))
        summary["total"] = total
        summary["n"] = total
        ok_n = int(summary.get("ok", 0))
        summary["ok"] = ok_n
        summary["fail"] = int(max(1, total - ok_n))
    return summary


def _run_materials_property_gate(*, preset: dict, out_dir: str) -> dict[str, object]:
    fixture = str(
        preset.get("materials_fixture", "examples/interop/materials_parity_suite_v2.json")
    )
    case_prefix = str(preset.get("case_prefix", ""))
    timeout = int(preset.get("timeout", 300))
    out_json = os.path.join(out_dir, "materials_property_gate.summary.json")
    cmd = [
        sys.executable,
        "scripts/materials_property_gate.py",
        "--fixture",
        fixture,
        "--config",
        "examples/td_1d_morse.yaml",
        "--out",
        out_json,
        "--strict",
    ]
    if case_prefix:
        cmd.extend(["--case-prefix", case_prefix])
    timed_out = False
    proc_rc = 1
    proc_out = ""
    proc_err = ""
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT_DIR, capture_output=True, text=True, timeout=max(1, timeout)
        )
        proc_rc = int(proc.returncode)
        proc_out = str(proc.stdout or "")
        proc_err = str(proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        proc_rc = 124
        proc_out = str(getattr(exc, "stdout", "") or "")
        proc_err = str(getattr(exc, "stderr", "") or "")

    summary: dict[str, object] = {
        "n": 0,
        "total": 0,
        "ok": 0,
        "fail": 1,
        "ok_all": False,
        "worst": {},
        "by_case": {},
        "external_script": "scripts/materials_property_gate.py",
        "external_returncode": int(proc_rc),
        "external_timed_out": bool(timed_out),
        "external_stdout": proc_out,
        "external_stderr": proc_err,
        "materials_fixture": fixture,
        "case_prefix": case_prefix,
        "external_artifacts": {"json": out_json},
    }
    if os.path.exists(out_json):
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                total = int(loaded.get("total", 0))
                ok_n = int(loaded.get("ok", 0))
                fail_n = int(loaded.get("fail", max(0, total - ok_n)))
                by_case = {}
                worst = {
                    "max_abs_diff": 0.0,
                    "property_fail": 0,
                }
                for c in list(loaded.get("cases", [])):
                    if not isinstance(c, dict):
                        continue
                    name = str(c.get("name", "unknown"))
                    by_case[name] = {
                        "total": 1,
                        "ok": int(bool(c.get("ok", False))),
                        "fail": int(not bool(c.get("ok", False))),
                        "worst": {
                            "max_abs_diff": float(c.get("max_abs_diff", 0.0)),
                            "property_fail": int(c.get("property_fail", 0)),
                        },
                    }
                    worst["max_abs_diff"] = max(
                        worst["max_abs_diff"], float(c.get("max_abs_diff", 0.0))
                    )
                    worst["property_fail"] = max(
                        worst["property_fail"], int(c.get("property_fail", 0))
                    )
                summary.update(
                    {
                        "total": total,
                        "n": total,
                        "ok": ok_n,
                        "fail": fail_n,
                        "ok_all": bool(loaded.get("ok_all", False)),
                        "worst": worst,
                        "by_case": by_case,
                        "by_property": dict(loaded.get("by_property", {}) or {}),
                    }
                )
        except Exception as exc:
            summary["parse_error"] = str(exc)
    summary["ok_all"] = bool(summary.get("ok_all", False) and int(proc_rc) == 0 and not timed_out)
    return summary


def _run_gpu_perf_smoke(*, preset: dict, out_dir: str) -> dict[str, object]:
    timeout = int(preset.get("timeout", 180))
    out_json = os.path.join(out_dir, "gpu_perf_smoke.summary.json")
    cmd = [
        sys.executable,
        "scripts/bench_gpu_perf_smoke.py",
        "--n-atoms",
        str(int(preset.get("n_atoms", 65536))),
        "--delta-count",
        str(int(preset.get("delta_count", 256))),
        "--repeats",
        str(int(preset.get("repeats", 9))),
        "--max-delta-over-full",
        str(float(preset.get("max_delta_over_full", 0.65))),
        "--max-transfer-over-kernel",
        str(float(preset.get("max_transfer_over_kernel", 4.0))),
        "--out-json",
        out_json,
        "--strict",
    ]
    timed_out = False
    proc_rc = 1
    proc_out = ""
    proc_err = ""
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT_DIR, capture_output=True, text=True, timeout=max(1, timeout)
        )
        proc_rc = int(proc.returncode)
        proc_out = str(proc.stdout or "")
        proc_err = str(proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        proc_rc = 124
        proc_out = str(getattr(exc, "stdout", "") or "")
        proc_err = str(getattr(exc, "stderr", "") or "")

    summary: dict[str, object] = {
        "n": 1,
        "total": 1,
        "ok": 0,
        "fail": 1,
        "ok_all": False,
        "worst": {},
        "by_case": {},
        "rows": [],
        "external_script": "scripts/bench_gpu_perf_smoke.py",
        "external_returncode": int(proc_rc),
        "external_timed_out": bool(timed_out),
        "external_stdout": proc_out,
        "external_stderr": proc_err,
        "external_artifacts": {"json": out_json},
    }
    if os.path.exists(out_json):
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                summary.update(loaded)
        except Exception as exc:
            summary["parse_error"] = str(exc)
    summary["ok_all"] = bool(summary.get("ok_all", False) and int(proc_rc) == 0 and not timed_out)
    summary["ok"] = int(bool(summary.get("ok_all", False)))
    summary["fail"] = int(not bool(summary.get("ok_all", False)))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml", help="Path to YAML config (examples/*.yaml)")
    ap.add_argument("--preset", choices=sorted(PRESETS.keys()), default="smoke_ci")
    ap.add_argument("--outdir", default="results", help="Root output directory")
    ap.add_argument("--strict", action="store_true", help="fail run if any row is not ok")
    ap.add_argument(
        "--golden", choices=['none', 'write', 'check'], default='none', help='golden baseline mode'
    )
    ap.add_argument("--cases-mode", choices=["cfg", "testcases"], default="cfg")
    ap.add_argument("--task", default="", help="Task YAML (for interop_smoke)")
    ap.add_argument(
        "--require-effective-cuda",
        action="store_true",
        help="fail if requested CUDA resolves to CPU fallback",
    )
    ap.add_argument(
        "--envelope-file", default="", help="Override envelope baseline file for supported presets"
    )
    ap.add_argument("--run-id", default=None, help="Optional run id (default timestamp)")
    args = ap.parse_args()

    cfg = load_config(args.yaml)
    p = PRESETS[args.preset]
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(args.outdir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    tol = dict(DEFAULT_THRESHOLDS)
    if isinstance(p.get("tol"), dict):
        tol.update(p.get("tol", {}))
    strict_guardrails = bool(p.get("strict_guardrails", False) or args.strict)
    requested_device = str(p.get("device", "cpu")).strip().lower()
    backend = _backend_evidence(requested_device)
    require_effective_cuda = bool(
        args.require_effective_cuda or p.get("require_effective_cuda", False)
    )

    rows = []
    mode_summary = None
    mode_kind = ""
    envelope_summary = None
    if bool(p.get("mpi_overlap_mode", False)):
        mode_summary = _run_mpi_overlap_sweep(preset=p, out_dir=out_dir)
        mode_kind = "mpi_overlap"
    elif bool(p.get("cluster_scale_mode", False)):
        mode_summary = _run_external_cluster_script(
            script_rel="scripts/bench_cluster_scale.py", preset=p, out_dir=out_dir
        )
        mode_kind = "cluster_scale"
    elif bool(p.get("cluster_stability_mode", False)):
        mode_summary = _run_external_cluster_script(
            script_rel="scripts/bench_cluster_stability.py", preset=p, out_dir=out_dir
        )
        mode_kind = "cluster_stability"
    elif bool(p.get("mpi_transport_matrix_mode", False)):
        mode_summary = _run_external_cluster_script(
            script_rel="scripts/bench_mpi_transport_matrix.py", preset=p, out_dir=out_dir
        )
        mode_kind = "mpi_transport_matrix"
    elif bool(p.get("viz_contract_mode", False)):
        mode_summary = _run_external_cluster_script(
            script_rel="scripts/bench_viz_contract.py", preset=p, out_dir=out_dir
        )
        mode_kind = "viz_contract"
    elif bool(p.get("materials_property_mode", False)):
        mode_summary = _run_materials_property_gate(preset=p, out_dir=out_dir)
        mode_kind = "materials_property"
    elif bool(p.get("gpu_perf_mode", False)):
        mode_summary = _run_gpu_perf_smoke(preset=p, out_dir=out_dir)
        mode_kind = "gpu_perf"
    elif bool(p.get("task_mode", False)):
        task_path = args.task or str(p.get("task_path", "examples/interop/task.yaml"))
        rows = sweep_verify_task(
            task_path,
            cfg.td,
            steps=int(p["steps"]),
            every=int(p["every"]),
            zones_total_list=list(p["zones_total_list"]),
            use_verlet_list=list(p["use_verlet_list"]),
            verlet_k_steps_list=list(p["verlet_k_steps_list"]),
            chaos_mode_list=list(p["chaos_mode_list"]),
            chaos_delay_prob_list=list(p["chaos_delay_prob_list"]),
            tol=tol,
            sync_mode=bool(p.get("sync_mode", False)),
            device=requested_device,
            strict_min_zone_width=bool(strict_guardrails),
        )
    else:
        pot = make_potential(cfg.potential.kind, cfg.potential.params)
        rows = sweep_verify2(
            cfg,
            pot,
            steps=int(p["steps"]),
            every=int(p["every"]),
            zones_total_list=list(p["zones_total_list"]),
            use_verlet_list=list(p["use_verlet_list"]),
            verlet_k_steps_list=list(p["verlet_k_steps_list"]),
            chaos_mode_list=list(p["chaos_mode_list"]),
            chaos_delay_prob_list=list(p["chaos_delay_prob_list"]),
            tol=tol,
            cases_mode=str(args.cases_mode),
            sync_mode=bool(p.get("sync_mode", False)),
            device=requested_device,
            strict_min_zone_width=bool(strict_guardrails),
        )

    backend_ok = (not require_effective_cuda) or bool(backend.get("ok_effective_cuda", False))
    if require_effective_cuda and not backend_ok:
        for r in rows:
            r.ok = False
            details = dict(r.details or {})
            violations = list(details.get("violations", []))
            violations.append("backend_fallback_cuda")
            details["backend"] = dict(backend)
            details["violations"] = violations
            r.details = details
    envelope_file = str(args.envelope_file).strip() or str(p.get("envelope_file", "")).strip()
    if rows and envelope_file:
        envelope_summary = apply_envelope_gate(rows, envelope_file=envelope_file)

    # artifacts
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "yaml": args.yaml,
                "preset": args.preset,
                "params": p,
                "cases_mode": args.cases_mode,
                "task": (args.task or ""),
                "backend": backend,
                "envelope_file": envelope_file,
                "envelope": envelope_summary,
                "strict_guardrails": strict_guardrails,
                "require_effective_cuda": require_effective_cuda,
            },
            f,
            indent=2,
        )
    if rows:
        write_csv(rows, os.path.join(out_dir, "metrics.csv"))
    summ = dict(mode_summary) if mode_summary is not None else summarize(rows)
    summ["backend"] = dict(backend)
    summ["strict_guardrails"] = bool(strict_guardrails)
    summ["require_effective_cuda"] = bool(require_effective_cuda)
    summ["backend_ok"] = bool(backend_ok)
    if envelope_summary is not None:
        summ["envelope"] = envelope_summary
    if mode_summary is not None:
        ok = bool(summ.get("ok_all", False)) and bool(backend_ok)
    else:
        ok = (all(r.ok for r in rows) if rows else False) and bool(backend_ok)
    # golden baseline (cfg_system only)
    if str(args.cases_mode) == "cfg":
        golden_path = os.path.join('golden', f'cfg_system_{args.preset}.json')
    else:
        golden_path = os.path.join('golden', f'{args.cases_mode}_{args.preset}.json')
    if args.golden in ('write', 'check'):
        os.makedirs('golden', exist_ok=True)
        key_rows = (
            [r for r in rows if r.case == 'cfg_system']
            if str(args.cases_mode) == "cfg"
            else list(rows)
        )
        if key_rows:
            g = {
                'preset': args.preset,
                'cases_mode': str(args.cases_mode),
                'rows': [
                    dict(
                        case=r.case,
                        zones_total=r.zones_total,
                        use_verlet=r.use_verlet,
                        verlet_k_steps=r.verlet_k_steps,
                        chaos_mode=r.chaos_mode,
                        chaos_delay_prob=r.chaos_delay_prob,
                        max_dr=r.max_dr,
                        max_dv=r.max_dv,
                        max_dE=r.max_dE,
                        max_dT=r.max_dT,
                        max_dP=r.max_dP,
                        ok=r.ok,
                    )
                    for r in key_rows
                ],
            }
            if args.golden == 'write':
                with open(golden_path, 'w', encoding='utf-8') as f:
                    json.dump(g, f, indent=2)
            else:
                if not os.path.exists(golden_path):
                    raise SystemExit(f'Golden file not found: {golden_path}')
                with open(golden_path, 'r', encoding='utf-8') as f:
                    g0 = json.load(f)

                # strict compare of ok flags and metric upper bounds
                def _row_key(x):
                    return (
                        x['case'],
                        x['zones_total'],
                        x['use_verlet'],
                        x['verlet_k_steps'],
                        x['chaos_mode'],
                        x['chaos_delay_prob'],
                    )

                base_map = {_row_key(x): x for x in g0.get('rows', [])}
                for r in key_rows:
                    k = (
                        r.case,
                        r.zones_total,
                        r.use_verlet,
                        r.verlet_k_steps,
                        r.chaos_mode,
                        r.chaos_delay_prob,
                    )
                    if k not in base_map:
                        raise SystemExit(f'Golden missing row {k}')
                    b = base_map[k]
                    if bool(r.ok) != bool(b['ok']):
                        raise SystemExit(
                            f'Golden ok mismatch for {k}: got {r.ok}, expected {b["ok"]}'
                        )
                    # metrics should not get worse by >10x (conservative)
                    for nm in ['max_dr', 'max_dv', 'max_dE', 'max_dT', 'max_dP']:
                        if float(r.__dict__[nm]) > 10.0 * float(b[nm]) + 1e-18:
                            raise SystemExit(
                                f'Golden regression {nm} for {k}: got {r.__dict__[nm]} expected <= {10.0*b[nm]}'
                            )
    incident_bundle = None
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2)
    if args.strict and (not ok):
        incident_bundle = {
            "path": write_incident_bundle(
                out_dir,
                reason="strict_failure",
                extra={
                    "preset": str(args.preset),
                    "strict": bool(args.strict),
                    "backend": dict(backend),
                    "backend_ok": bool(backend_ok),
                    "envelope": envelope_summary,
                    "mode_kind": str(mode_kind),
                    "external_mode": bool(mode_summary is not None),
                },
            ),
            "reason": "strict_failure",
        }
        summ["incident_bundle"] = incident_bundle
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summ, f, indent=2)

    if mode_summary is not None:
        ok = bool(summ.get("ok_all", False)) and bool(backend_ok)
    else:
        ok = (all(r.ok for r in rows) if rows else False) and bool(backend_ok)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(f"# VerifyLab run {run_id}\n\n")
        f.write(f"- preset: `{args.preset}`\n")
        f.write(f"- ok: `{ok}`\n")
        f.write(
            f"- backend: requested=`{backend['requested_device']}` "
            f"effective=`{backend['effective_device']}` "
            f"fallback_from_cuda=`{backend['fallback_from_cuda']}`\n"
        )
        f.write(f"- strict_guardrails: `{strict_guardrails}`\n")
        f.write(f"- require_effective_cuda: `{require_effective_cuda}`\n")
        if envelope_summary is not None:
            f.write(f"- envelope_ok: `{bool(envelope_summary.get('ok_all', False))}`\n")
        f.write("\n## Worst metrics\n\n")
        if 'worst' in summ:
            f.write(json.dumps(summ['worst'], indent=2))
            f.write("\n")
        f.write("\n## Summary\n\n")
        if mode_summary is not None:
            if mode_kind == "mpi_overlap":
                for rr in list(summ.get("mpi_overlap_runs", [])):
                    f.write(
                        f"- ranks={int(rr.get('ranks', 0))} ok={bool(rr.get('ok', False))} "
                        f"speedup={float(rr.get('overlap_speedup', 0.0)):.6f} "
                        f"hG={int(rr.get('hG_max', 0))} hV={int(rr.get('hV_max', 0))} "
                        f"violW={int(rr.get('violW_max', 0))} lagV={int(rr.get('lagV_max', 0))} "
                        f"asyncS={int(rr.get('async_send_msgs_max', 0))} "
                        f"asyncB={int(rr.get('async_send_bytes_max', 0))} "
                        f"wfgC={int(rr.get('wfgC_max', 0))} wfgO={int(rr.get('wfgO_max', 0))} "
                        f"rate={float(rr.get('wfgC_rate', 0.0)):.3f} "
                        f"p100={float(rr.get('wfgC_per_100_steps', 0.0)):.3f}\n"
                    )

                # WFG diagnostics warning (non-fatal): cycles are allowed under A4b, but indicate contention.
                runs = list(summ.get("mpi_overlap_runs", []))
                wfgC = [int(r.get("wfgC_max", 0)) for r in runs]
                wfgO = [int(r.get("wfgO_max", 0)) for r in runs]
                wfgS = [int(r.get("wfgS_max", 0)) for r in runs]
                if any(v > 0 for v in wfgC):
                    worstC = max(wfgC) if wfgC else 0
                    worstO = max(wfgO) if wfgO else 0
                    # ranks where max occurs (mpi overlap already grouped by nranks, still show it for completeness)
                    ranks_at_worst = [
                        int(r.get("ranks", -1)) for r in runs if int(r.get("wfgC_max", 0)) == worstC
                    ]
                    ranks_at_worst = sorted(set([r for r in ranks_at_worst if r >= 0]))
                    # empirical cycle rate proxy: max(wfgC)/max(wfgS) (local sampling)
                    rate = 0.0
                    if max(wfgS) > 0:
                        rate = float(worstC) / float(max(wfgS))
                    f.write("\n### WFG contention diagnostics (mpi_overlap)\n")
                    f.write(
                        f"**WARNING (non-fatal):** transient local WFG cycles observed under A4b. "
                        f"max(wfgC)={worstC}, max(wfgO)={worstO}, cycle_rate≈{rate:.3f} per sample.\n"
                    )
                    if ranks_at_worst:
                        f.write(f"- Observed worst cycles at ranks={ranks_at_worst}\n")
                    f.write(
                        "- Interpretation: A4b prevents deadlock; wfgC>0 indicates contention/near-cycles in local wait patterns.\n"
                        "- Next steps: inspect overlap/timing knobs and consult `docs/WFG_DIAGNOSTICS.md` for how to localize donors and hot spots.\n"
                    )
                    f.write("\n")
            elif mode_kind == "materials_property":
                for cname, csum in dict(summ.get("by_case", {}) or {}).items():
                    if not isinstance(csum, dict):
                        continue
                    worst = dict(csum.get("worst", {}) or {})
                    f.write(
                        f"- case={cname} ok={int(csum.get('ok', 0))}/{int(csum.get('total', 0))} "
                        f"max_abs_diff={float(worst.get('max_abs_diff', 0.0)):.6e} "
                        f"property_fail={int(worst.get('property_fail', 0))}\n"
                    )
                byp = dict(summ.get("by_property", {}) or {})
                if byp:
                    f.write("\n")
                    for grp, st in sorted(byp.items()):
                        if not isinstance(st, dict):
                            continue
                        f.write(
                            f"- property_group={grp} ok={int(st.get('ok', 0))} fail={int(st.get('fail', 0))} "
                            f"max_abs_diff={float(st.get('max_abs_diff', 0.0)):.6e}\n"
                        )
            else:
                for rr in list(summ.get("rows", []))[:20]:
                    if not isinstance(rr, dict):
                        continue
                    line = (
                        f"- mode={mode_kind} "
                        f"ranks={int(rr.get('ranks', 0))} "
                        f"ok={bool(rr.get('ok', False))} "
                        f"elapsed={float(rr.get('elapsed_sec', 0.0)):.6f}s "
                        f"hG={int(rr.get('hG_max', 0))} hV={int(rr.get('hV_max', 0))} "
                        f"violW={int(rr.get('violW_max', 0))} lagV={int(rr.get('lagV_max', 0))}"
                    )
                    if "speedup_vs_blocking" in rr:
                        line += f" speedup={float(rr.get('speedup_vs_blocking', 0.0)):.6f}"
                    elif "speedup" in rr:
                        line += f" speedup={float(rr.get('speedup', 0.0)):.6f}"
                    if "efficiency" in rr:
                        line += f" eff={float(rr.get('efficiency', 0.0)):.6f}"
                    f.write(line + "\n")
        else:
            f.write(summarize_markdown(rows))
        if require_effective_cuda and not backend_ok:
            f.write("\n\n## Backend gate\n\n")
            f.write("- violation: `backend_fallback_cuda`\n")
            f.write("- CUDA was requested but effective backend is CPU.\n")
        if incident_bundle is not None:
            f.write("\n\n## Incident bundle\n\n")
            f.write(f"- path: `{incident_bundle['path']}`\n")
            f.write(f"- reason: `{incident_bundle['reason']}`\n")
        if envelope_summary is not None:
            f.write("\n\n## Envelope gate\n\n")
            f.write(f"- file: `{envelope_summary.get('file', '')}`\n")
            f.write(f"- ok_all: `{bool(envelope_summary.get('ok_all', False))}`\n")
            f.write(f"- rows_checked: `{int(envelope_summary.get('rows_checked', 0))}`\n")
            f.write(f"- rows_failed: `{int(envelope_summary.get('rows_failed', 0))}`\n")
            f.write(
                f"- missing_baseline_rows: `{int(envelope_summary.get('missing_baseline_rows', 0))}`\n"
            )
        f.write('\n\n```json\n')
        f.write(json.dumps(summ, indent=2))
        f.write('\n```\n')
        f.write("\n")

    if args.strict and (not ok):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
