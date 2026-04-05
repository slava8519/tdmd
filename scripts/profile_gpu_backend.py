from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PHASE_E_COMMIT = "efb864e"
DEFAULT_PHASE_E_LABEL = "Phase E CuPy high-level baseline before PR-C02"

_PHASE_E_RUNNER = textwrap.dedent(
    """
    import runpy
    import sys

    repo = sys.argv[1]
    script = sys.argv[2]
    args = sys.argv[3:]
    sys.path.insert(0, repo)
    sys.argv = [script] + args
    runpy.run_path(script, run_name="__main__")
    """
).strip()


def _run(
    cmd: list[str],
    *,
    cwd: str,
    timeout: int,
    env: dict[str, str] | None = None,
) -> tuple[int, float, str, str]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    dt = float(time.perf_counter() - t0)
    return int(proc.returncode), dt, str(proc.stdout or ""), str(proc.stderr or "")


def _read_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _ratio(numer: float, denom: float) -> float | None:
    numer_f = float(numer)
    denom_f = float(denom)
    if numer_f <= 0.0 or denom_f <= 0.0:
        return None
    return float(numer_f / denom_f)


def _fmt_ratio(value: float | None) -> str:
    return f"{float(value):.3f}x" if value is not None else "n/a"


def _fmt_time(value: float | None) -> str:
    if value is None or float(value) <= 0.0:
        return "n/a"
    return f"{float(value):.6f}s"


def _write_verify_csv(rows: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "preset",
                "run_id",
                "returncode",
                "elapsed_sec",
                "ok_all",
                "max_dr",
                "max_dv",
                "max_dE",
                "max_dT",
                "max_dP",
                "stdout_bytes",
                "stderr_bytes",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["preset"],
                    r["run_id"],
                    r["returncode"],
                    f"{float(r['elapsed_sec']):.6f}",
                    int(bool(r["ok_all"])),
                    r["max_dr"],
                    r["max_dv"],
                    r["max_dE"],
                    r["max_dT"],
                    r["max_dP"],
                    r["stdout_bytes"],
                    r["stderr_bytes"],
                ]
            )


def _run_verify_presets(*, cfg: str, timeout: int) -> tuple[list[dict[str, object]], dict[str, float]]:
    rows: list[dict[str, object]] = []
    runs = [
        ("smoke_ci", "profile_cpu_smoke"),
        ("gpu_smoke", "profile_gpu_smoke"),
        ("interop_smoke", "profile_cpu_interop"),
        ("gpu_interop_smoke", "profile_gpu_interop"),
        ("interop_metal_smoke", "profile_cpu_metal"),
        ("gpu_metal_smoke", "profile_gpu_metal"),
    ]
    for preset, run_id in runs:
        cmd = [
            sys.executable,
            "scripts/run_verifylab_matrix.py",
            cfg,
            "--preset",
            preset,
            "--strict",
            "--run-id",
            run_id,
        ]
        rc, elapsed, out, err = _run(cmd, cwd=str(ROOT_DIR), timeout=int(timeout))
        summary = _read_summary(ROOT_DIR / "results" / run_id / "summary.json")
        worst = summary.get("worst", {}) if isinstance(summary, dict) else {}
        rows.append(
            {
                "preset": preset,
                "run_id": run_id,
                "returncode": rc,
                "elapsed_sec": elapsed,
                "ok_all": bool(summary.get("ok_all", False)) if isinstance(summary, dict) else False,
                "max_dr": _safe_float(worst.get("max_dr", 0.0)),
                "max_dv": _safe_float(worst.get("max_dv", 0.0)),
                "max_dE": _safe_float(worst.get("max_dE", 0.0)),
                "max_dT": _safe_float(worst.get("max_dT", 0.0)),
                "max_dP": _safe_float(worst.get("max_dP", 0.0)),
                "stdout_bytes": len(out.encode("utf-8")),
                "stderr_bytes": len(err.encode("utf-8")),
            }
        )

    base = {str(r["preset"]): float(r["elapsed_sec"]) for r in rows}
    ratios = {
        "smoke_cpu_over_gpu": _ratio(base.get("smoke_ci", 0.0), base.get("gpu_smoke", 0.0)),
        "interop_cpu_over_gpu": _ratio(
            base.get("interop_smoke", 0.0), base.get("gpu_interop_smoke", 0.0)
        ),
        "metal_cpu_over_gpu": _ratio(
            base.get("interop_metal_smoke", 0.0), base.get("gpu_metal_smoke", 0.0)
        ),
    }
    return rows, ratios


def _run_gpu_perf_smoke(*, out_dir: Path, timeout: int, require_effective_cuda: bool) -> dict[str, object]:
    out_json = out_dir / "gpu_perf_smoke.summary.json"
    cmd = [
        sys.executable,
        "scripts/bench_gpu_perf_smoke.py",
        "--out-json",
        str(out_json),
    ]
    if require_effective_cuda:
        cmd.append("--strict")
    rc, elapsed, out, err = _run(cmd, cwd=str(ROOT_DIR), timeout=int(timeout))
    summary = _read_summary(out_json)
    return {
        "returncode": int(rc),
        "elapsed_sec": float(elapsed),
        "stdout": out,
        "stderr": err,
        "summary": summary,
        "artifact_json": str(out_json),
    }


def _run_eam_benchmark_current(
    *,
    out_dir: Path,
    timeout: int,
    require_effective_cuda: bool,
    bench_args: list[str],
) -> dict[str, object]:
    out_csv = out_dir / "eam_decomp_perf.csv"
    out_md = out_dir / "eam_decomp_perf.md"
    out_json = out_dir / "eam_decomp_perf.summary.json"
    cmd = [
        sys.executable,
        "scripts/bench_eam_decomp_perf.py",
        "--out",
        str(out_csv),
        "--md",
        str(out_md),
        "--json",
        str(out_json),
    ]
    cmd.extend(bench_args)
    if require_effective_cuda:
        cmd.extend(["--require-effective-cuda", "--strict"])
    rc, elapsed, out, err = _run(cmd, cwd=str(ROOT_DIR), timeout=int(timeout))
    summary = _read_summary(out_json)
    return {
        "returncode": int(rc),
        "elapsed_sec": float(elapsed),
        "stdout": out,
        "stderr": err,
        "summary": summary,
        "artifacts": {
            "csv": str(out_csv),
            "md": str(out_md),
            "json": str(out_json),
        },
    }


def _run_eam_benchmark_phase_e(
    *,
    phase_e_worktree: str,
    out_dir: Path,
    timeout: int,
    require_effective_cuda: bool,
    bench_args: list[str],
) -> dict[str, object]:
    out_csv = out_dir / "eam_decomp_perf_phase_e.csv"
    out_md = out_dir / "eam_decomp_perf_phase_e.md"
    out_json = out_dir / "eam_decomp_perf_phase_e.summary.json"
    cmd = [
        sys.executable,
        "-c",
        _PHASE_E_RUNNER,
        str(phase_e_worktree),
        str(ROOT_DIR / "scripts" / "bench_eam_decomp_perf.py"),
        "--out",
        str(out_csv),
        "--md",
        str(out_md),
        "--json",
        str(out_json),
    ]
    cmd.extend(bench_args)
    if require_effective_cuda:
        cmd.extend(["--require-effective-cuda", "--strict"])
    rc, elapsed, out, err = _run(cmd, cwd=str(ROOT_DIR), timeout=int(timeout))
    summary = _read_summary(out_json)
    return {
        "returncode": int(rc),
        "elapsed_sec": float(elapsed),
        "stdout": out,
        "stderr": err,
        "summary": summary,
        "artifacts": {
            "csv": str(out_csv),
            "md": str(out_md),
            "json": str(out_json),
        },
    }


def _phase_e_comparison(
    *,
    current_summary: dict,
    phase_e_summary: dict,
) -> dict[str, object]:
    cur_cases = dict(current_summary.get("by_case", {}) or {})
    old_cases = dict(phase_e_summary.get("by_case", {}) or {})
    out: dict[str, object] = {}
    for case in ("space_gpu", "time_gpu"):
        cur = dict(cur_cases.get(case, {}) or {})
        old = dict(old_cases.get(case, {}) or {})
        out[f"{case}_speedup_vs_phase_e"] = _ratio(
            _safe_float(old.get("elapsed_sec_median", 0.0)),
            _safe_float(cur.get("elapsed_sec_median", 0.0)),
        )
    return out


def _plan_b_assessment(
    *,
    verify_ratios: dict[str, float | None],
    gpu_perf_summary: dict,
    eam_current_summary: dict,
    phase_e_summary: dict | None,
    phase_e_compare: dict[str, object] | None,
) -> dict[str, object]:
    reasons: list[str] = []
    warnings: list[str] = []

    smoke_ratio = verify_ratios.get("smoke_cpu_over_gpu")
    metal_ratio = verify_ratios.get("metal_cpu_over_gpu")
    perf_worst = dict(gpu_perf_summary.get("worst", {}) or {})
    transfer_over_kernel = _safe_float(perf_worst.get("transfer_over_kernel", 0.0))
    delta_over_full = _safe_float(perf_worst.get("delta_over_full", 0.0))

    eam_cmp = dict(eam_current_summary.get("comparisons", {}) or {})
    time_gpu_vs_cpu = eam_cmp.get("gpu_speedup_time")
    space_gpu_vs_cpu = eam_cmp.get("gpu_speedup_space")

    if smoke_ratio is not None and smoke_ratio > 1.0:
        reasons.append(f"gpu_smoke faster than smoke_ci ({_fmt_ratio(smoke_ratio)})")
    else:
        warnings.append("gpu_smoke did not outperform smoke_ci")

    if metal_ratio is not None and metal_ratio > 1.0:
        reasons.append(f"gpu_metal_smoke faster than interop_metal_smoke ({_fmt_ratio(metal_ratio)})")
    else:
        warnings.append("gpu_metal_smoke did not outperform interop_metal_smoke")

    if time_gpu_vs_cpu is not None and float(time_gpu_vs_cpu) > 1.0:
        reasons.append(f"EAM time_gpu beats time_cpu ({_fmt_ratio(time_gpu_vs_cpu)})")
    else:
        warnings.append("EAM time_gpu does not beat time_cpu")

    if space_gpu_vs_cpu is not None and float(space_gpu_vs_cpu) > 1.0:
        reasons.append(f"EAM space_gpu beats space_cpu ({_fmt_ratio(space_gpu_vs_cpu)})")
    else:
        warnings.append("EAM space_gpu does not beat space_cpu")

    if transfer_over_kernel > 0.0 and transfer_over_kernel <= 4.0:
        reasons.append(f"transfer_over_kernel within perf-smoke envelope ({transfer_over_kernel:.3f})")
    elif transfer_over_kernel > 0.0:
        warnings.append(f"transfer_over_kernel exceeds perf-smoke envelope ({transfer_over_kernel:.3f})")

    if delta_over_full > 0.0 and delta_over_full <= 0.65:
        reasons.append(f"delta_over_full within perf-smoke envelope ({delta_over_full:.3f})")
    elif delta_over_full > 0.0:
        warnings.append(f"delta_over_full exceeds perf-smoke envelope ({delta_over_full:.3f})")

    if phase_e_summary and phase_e_compare:
        time_vs_phase_e = phase_e_compare.get("time_gpu_speedup_vs_phase_e")
        space_vs_phase_e = phase_e_compare.get("space_gpu_speedup_vs_phase_e")
        if time_vs_phase_e is not None and float(time_vs_phase_e) > 1.0:
            reasons.append(f"time_gpu improves over Phase E ({_fmt_ratio(time_vs_phase_e)})")
        else:
            warnings.append("time_gpu does not improve over Phase E baseline")
        if space_vs_phase_e is not None and float(space_vs_phase_e) > 1.0:
            reasons.append(f"space_gpu improves over Phase E ({_fmt_ratio(space_vs_phase_e)})")
        else:
            warnings.append("space_gpu does not improve over Phase E baseline")
    else:
        warnings.append("Phase E comparison unavailable")

    recommend_plan_b = bool(
        (time_gpu_vs_cpu is not None and float(time_gpu_vs_cpu) <= 1.0)
        or (
            phase_e_compare is not None
            and phase_e_compare.get("time_gpu_speedup_vs_phase_e") is not None
            and float(phase_e_compare["time_gpu_speedup_vs_phase_e"]) <= 1.0
            and transfer_over_kernel > 4.0
        )
        or (
            phase_e_compare is None
            and transfer_over_kernel > 5.0
        )
    )
    return {
        "recommendation": ("evaluate_cpp_cuda_extension" if recommend_plan_b else "stay_on_rawkernel"),
        "reasons": reasons,
        "warnings": warnings,
        "phase_e_reference_commit": DEFAULT_PHASE_E_COMMIT,
        "phase_e_reference_label": DEFAULT_PHASE_E_LABEL,
    }


def _build_markdown(
    *,
    cfg: str,
    eam_bench_args: list[str],
    verify_rows: list[dict[str, object]],
    verify_ratios: dict[str, float | None],
    gpu_perf_summary: dict,
    gpu_perf_artifact: str,
    eam_current_summary: dict,
    eam_current_artifacts: dict[str, str],
    phase_e_worktree: str,
    phase_e_summary: dict | None,
    phase_e_compare: dict[str, object] | None,
    plan_b: dict[str, object],
) -> str:
    lines = [
        "# GPU Backend Profile",
        "",
        f"- config: `{cfg}`",
        f"- smoke cpu/gpu ratio: `{_fmt_ratio(verify_ratios.get('smoke_cpu_over_gpu'))}`",
        f"- interop cpu/gpu ratio: `{_fmt_ratio(verify_ratios.get('interop_cpu_over_gpu'))}`",
        f"- metal cpu/gpu ratio: `{_fmt_ratio(verify_ratios.get('metal_cpu_over_gpu'))}`",
        f"- phase-e reference commit: `{DEFAULT_PHASE_E_COMMIT}`",
        f"- phase-e reference label: `{DEFAULT_PHASE_E_LABEL}`",
        f"- eam bench args: `{' '.join(eam_bench_args) if eam_bench_args else '(defaults)'}`",
        "",
        "## Verify Presets",
        "",
    ]
    for row in verify_rows:
        lines.append(
            f"- `{row['preset']}` rc={row['returncode']} ok={bool(row['ok_all'])} "
            f"elapsed={float(row['elapsed_sec']):.3f}s "
            f"max_dr={float(row['max_dr']):.3e} max_dv={float(row['max_dv']):.3e} "
            f"max_dE={float(row['max_dE']):.3e}"
        )

    worst = dict(gpu_perf_summary.get("worst", {}) or {})
    lines.extend(
        [
            "",
            "## GPU Perf Smoke",
            "",
            f"- artifact: `{gpu_perf_artifact}`",
            f"- ok_all: `{bool(gpu_perf_summary.get('ok_all', False))}`",
            f"- h2d_full_ms: `{_safe_float(worst.get('h2d_full_ms', 0.0)):.6f}`",
            f"- h2d_delta_ms: `{_safe_float(worst.get('h2d_delta_ms', 0.0)):.6f}`",
            f"- kernel_ms: `{_safe_float(worst.get('kernel_ms', 0.0)):.6f}`",
            f"- d2h_ms: `{_safe_float(worst.get('d2h_ms', 0.0)):.6f}`",
            f"- delta_over_full: `{_safe_float(worst.get('delta_over_full', 0.0)):.6f}`",
            f"- transfer_over_kernel: `{_safe_float(worst.get('transfer_over_kernel', 0.0)):.6f}`",
        ]
    )
    calibration = dict(gpu_perf_summary.get("calibration", {}) or {})
    if calibration:
        lines.extend(
            [
                f"- kernel_loop_iters: `{int(calibration.get('kernel_loop_iters', 0) or 0)}`",
                f"- calibration_applied: `{bool(calibration.get('applied', False))}`",
                f"- calibration_target_met: `{bool(calibration.get('target_met', False))}`",
                f"- min_kernel_ms: `{_safe_float(calibration.get('min_kernel_ms', 0.0)):.6f}`",
                f"- probe_kernel_ms: `{_safe_float(calibration.get('probe_kernel_ms', 0.0)):.6f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## EAM Decomposition",
            "",
            f"- artifacts: `csv={eam_current_artifacts['csv']}`, `md={eam_current_artifacts['md']}`, `json={eam_current_artifacts['json']}`",
            f"- ok_all: `{bool(eam_current_summary.get('ok_all', False))}`",
            f"- gpu_speedup_space: `{_fmt_ratio(dict(eam_current_summary.get('comparisons', {}) or {}).get('gpu_speedup_space'))}`",
            f"- gpu_speedup_time: `{_fmt_ratio(dict(eam_current_summary.get('comparisons', {}) or {}).get('gpu_speedup_time'))}`",
            f"- time_speedup_cpu: `{_fmt_ratio(dict(eam_current_summary.get('comparisons', {}) or {}).get('time_speedup_cpu'))}`",
            f"- time_speedup_gpu: `{_fmt_ratio(dict(eam_current_summary.get('comparisons', {}) or {}).get('time_speedup_gpu'))}`",
        ]
    )

    lines.extend(["", "## Phase E Comparison", ""])
    if phase_e_summary and phase_e_compare:
        lines.append(f"- worktree: `{phase_e_worktree}`")
        lines.append(f"- time_gpu_speedup_vs_phase_e: `{_fmt_ratio(phase_e_compare.get('time_gpu_speedup_vs_phase_e'))}`")
        lines.append(f"- space_gpu_speedup_vs_phase_e: `{_fmt_ratio(phase_e_compare.get('space_gpu_speedup_vs_phase_e'))}`")
    else:
        lines.append("- comparison unavailable; pass `--phase-e-worktree <path>` to benchmark the historical baseline")

    lines.extend(["", "## Plan B Decision", ""])
    lines.append(f"- recommendation: `{plan_b['recommendation']}`")
    for reason in list(plan_b.get("reasons", [])):
        lines.append(f"- reason: {reason}")
    for warning in list(plan_b.get("warnings", [])):
        lines.append(f"- warning: {warning}")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="Profile current GPU backend and optional Phase E baseline")
    p.add_argument("--config", default="examples/td_1d_morse.yaml")
    p.add_argument("--out-csv", default="results/gpu_profile.csv")
    p.add_argument("--out-md", default="results/gpu_profile.md")
    p.add_argument("--out-json", default="results/gpu_profile.summary.json")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--phase-e-worktree", default="")
    p.add_argument("--require-effective-cuda", action="store_true")
    p.add_argument("--eam-n-atoms", type=int, default=256)
    p.add_argument("--eam-steps", type=int, default=1)
    p.add_argument("--eam-repeats", type=int, default=1)
    p.add_argument("--eam-warmup", type=int, default=0)
    args = p.parse_args()

    cfg = str(args.config)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    out_dir = out_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    eam_bench_args = [
        "--n-atoms",
        str(int(args.eam_n_atoms)),
        "--steps",
        str(int(args.eam_steps)),
        "--repeats",
        str(int(args.eam_repeats)),
        "--warmup",
        str(int(args.eam_warmup)),
    ]

    verify_rows, verify_ratios = _run_verify_presets(cfg=cfg, timeout=int(args.timeout))
    _write_verify_csv(verify_rows, out_csv)

    gpu_perf = _run_gpu_perf_smoke(
        out_dir=out_dir,
        timeout=int(args.timeout),
        require_effective_cuda=bool(args.require_effective_cuda),
    )
    gpu_perf_summary = dict(gpu_perf.get("summary", {}) or {})

    eam_current = _run_eam_benchmark_current(
        out_dir=out_dir,
        timeout=int(args.timeout),
        require_effective_cuda=bool(args.require_effective_cuda),
        bench_args=eam_bench_args,
    )
    eam_current_summary = dict(eam_current.get("summary", {}) or {})

    phase_e_summary = None
    phase_e_compare = None
    if str(args.phase_e_worktree).strip():
        phase_e = _run_eam_benchmark_phase_e(
            phase_e_worktree=str(args.phase_e_worktree).strip(),
            out_dir=out_dir,
            timeout=int(args.timeout),
            require_effective_cuda=bool(args.require_effective_cuda),
            bench_args=eam_bench_args,
        )
        phase_e_summary = dict(phase_e.get("summary", {}) or {})
        phase_e_compare = _phase_e_comparison(
            current_summary=eam_current_summary,
            phase_e_summary=phase_e_summary,
        )

    plan_b = _plan_b_assessment(
        verify_ratios=verify_ratios,
        gpu_perf_summary=gpu_perf_summary,
        eam_current_summary=eam_current_summary,
        phase_e_summary=phase_e_summary,
        phase_e_compare=phase_e_compare,
    )

    report_md = _build_markdown(
        cfg=cfg,
        eam_bench_args=eam_bench_args,
        verify_rows=verify_rows,
        verify_ratios=verify_ratios,
        gpu_perf_summary=gpu_perf_summary,
        gpu_perf_artifact=str(gpu_perf.get("artifact_json", "")),
        eam_current_summary=eam_current_summary,
        eam_current_artifacts=dict(eam_current.get("artifacts", {}) or {}),
        phase_e_worktree=str(args.phase_e_worktree).strip(),
        phase_e_summary=phase_e_summary,
        phase_e_compare=phase_e_compare,
        plan_b=plan_b,
    )
    out_md.write_text(report_md, encoding="utf-8")

    summary = {
        "config": cfg,
        "verify_rows": verify_rows,
        "verify_ratios": verify_ratios,
        "eam_bench_args": eam_bench_args,
        "gpu_perf_smoke": gpu_perf,
        "eam_current": eam_current,
        "phase_e_reference": {
            "commit": DEFAULT_PHASE_E_COMMIT,
            "label": DEFAULT_PHASE_E_LABEL,
            "worktree": str(args.phase_e_worktree).strip(),
            "summary": phase_e_summary,
            "comparisons": phase_e_compare,
        },
        "plan_b_assessment": plan_b,
        "artifacts": {
            "csv": str(out_csv),
            "md": str(out_md),
            "json": str(out_json),
        },
        "ok_all": bool(
            all(bool(r.get("ok_all", False)) and int(r.get("returncode", 1)) == 0 for r in verify_rows)
            and bool(gpu_perf_summary.get("ok_all", not args.require_effective_cuda))
            and bool(eam_current_summary.get("ok_all", False) or not args.require_effective_cuda)
            and (
                phase_e_summary is None
                or bool(phase_e_summary.get("ok_all", False) or not args.require_effective_cuda)
            )
        ),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[gpu-profile] wrote {out_csv}")
    print(f"[gpu-profile] wrote {out_md}")
    print(f"[gpu-profile] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
