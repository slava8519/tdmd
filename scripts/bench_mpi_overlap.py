from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tdmd.cluster_profile import apply_profile_env, capture_env_snapshot, load_cluster_profile


def _find_mpirun(user_arg: str | None) -> str | None:
    if user_arg:
        return user_arg
    env_val = os.environ.get("MPIRUN", "").strip()
    if env_val:
        return env_val
    return shutil.which("mpiexec.hydra") or shutil.which("mpiexec") or shutil.which("mpirun")


def _parse_overlap_list(text: str) -> list[int]:
    vals: list[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        iv = int(token)
        if iv not in (0, 1):
            raise ValueError("overlap-list values must be 0 or 1")
        vals.append(iv)
    if not vals:
        raise ValueError("overlap-list must contain at least one value")
    return vals


def _write_cfg(
    base: dict,
    *,
    overlap: bool,
    cuda_aware_mpi: bool,
    strict_zone_width: bool,
    n_steps: int,
    thermo_every: int,
) -> str:
    cfg = dict(base)
    td = dict(cfg.get("td", {}))
    td["comm_overlap_isend"] = bool(overlap)
    td["cuda_aware_mpi"] = bool(cuda_aware_mpi)
    td["strict_min_zone_width"] = bool(strict_zone_width)
    cfg["td"] = td

    run = dict(cfg.get("run", {}))
    if int(n_steps) > 0:
        run["n_steps"] = int(n_steps)
    # Ensure periodic diagnostics are emitted so counters can be parsed.
    run["thermo_every"] = max(1, int(thermo_every))
    cfg["run"] = run

    fd, path = tempfile.mkstemp(prefix=f"tdmd_overlap_{int(overlap)}_", suffix=".yaml")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def _run(
    cmd: list[str], cwd: Path, timeout: int, env: dict[str, str]
) -> tuple[int, float, str, str]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=str(cwd), timeout=int(timeout), capture_output=True, text=True, env=env
    )
    return (
        int(proc.returncode),
        float(time.perf_counter() - t0),
        str(proc.stdout or ""),
        str(proc.stderr or ""),
    )


def _diag_max(text: str, key: str) -> tuple[int, int]:
    vals = [int(m.group(1)) for m in re.finditer(rf"\b{re.escape(key)}=(\d+)\b", str(text))]
    if not vals:
        return 0, 0
    return max(vals), len(vals)


def _diag_max_float(text: str, key: str) -> tuple[float, int]:
    vals = [
        float(m.group(1))
        for m in re.finditer(rf"\b{re.escape(key)}=([-+]?\d+(?:\.\d+)?)\b", str(text))
    ]
    if not vals:
        return 0.0, 0
    return max(vals), len(vals)


def _sim_elapsed(*, n_ranks: int, overlap: int, cuda_aware: bool) -> float:
    base = 12.0 / max(1.0, float(n_ranks))
    if int(overlap) == 1:
        base *= 0.90
    if bool(cuda_aware) and int(overlap) == 1:
        base *= 0.95
    return float(base)


def _iter_rows(
    *,
    overlap_modes: Iterable[int],
    base: dict,
    enable_cuda_aware_mpi: bool,
    mpirun: str,
    n_ranks: int,
    timeout: int,
    root: Path,
    strict_invariants: bool,
    require_async_evidence: bool,
    require_overlap_window: bool,
    retries: int,
    strict_zone_width: bool,
    n_steps: int,
    thermo_every: int,
    simulate: bool,
    run_env: dict[str, str],
) -> list[dict[str, int | float]]:
    rows: list[dict[str, int | float]] = []
    for overlap in overlap_modes:
        overlap_i = int(overlap)
        if bool(simulate):
            elapsed = _sim_elapsed(
                n_ranks=int(n_ranks), overlap=overlap_i, cuda_aware=bool(enable_cuda_aware_mpi)
            )
            async_msgs = int(max(0, n_ranks)) if overlap_i == 1 else 0
            async_bytes = int(async_msgs * 64)
            async_ok = bool((overlap_i == 0) or (not require_async_evidence) or (async_msgs > 0))
            send_pack_ms = float(0.08 if overlap_i == 1 else 0.12)
            send_wait_ms = float(0.04 if overlap_i == 1 else 0.09)
            recv_poll_ms = float(0.06 if overlap_i == 1 else 0.05)
            overlap_win_ms = float(0.22 if overlap_i == 1 else 0.0)
            overlap_win_ok = bool(
                (overlap_i == 0)
                or (not require_overlap_window)
                or (float(overlap_win_ms) > 0.0)
            )
            rows.append(
                {
                    "overlap": overlap_i,
                    "cuda_aware_mpi": int(bool(enable_cuda_aware_mpi and overlap_i == 1)),
                    "returncode": 0,
                    "elapsed_sec": float(elapsed),
                    "hG_max": 0,
                    "hV_max": 0,
                    "violW_max": 0,
                    "lagV_max": 0,
                    "async_send_msgs_max": int(async_msgs),
                    "async_send_bytes_max": int(async_bytes),
                    "async_evidence_ok": int(async_ok),
                    "send_pack_ms_max": float(send_pack_ms),
                    "send_wait_ms_max": float(send_wait_ms),
                    "recv_poll_ms_max": float(recv_poll_ms),
                    "overlap_window_ms_max": float(overlap_win_ms),
                    "overlap_window_ok": int(overlap_win_ok),
                    "diag_samples": max(1, int(thermo_every)),
                    "invariants_ok": 1,
                    "strict_invariants_ok": 1,
                    "attempts": 0,
                    "simulated": 1,
                }
            )
            continue

        cfg_tmp = _write_cfg(
            base,
            overlap=bool(overlap_i),
            cuda_aware_mpi=bool(enable_cuda_aware_mpi and overlap_i == 1),
            strict_zone_width=bool(strict_zone_width),
            n_steps=int(n_steps),
            thermo_every=int(thermo_every),
        )
        try:
            cmd = [
                str(mpirun),
                "-n",
                str(int(n_ranks)),
                sys.executable,
                "-m",
                "tdmd.main",
                "run",
                cfg_tmp,
            ]
            attempts = max(1, int(retries))
            rc = 1
            elapsed = 0.0
            out = ""
            err = ""
            for _ in range(1, attempts + 1):
                rc_try, elapsed_try, out_try, err_try = _run(
                    cmd, cwd=root, timeout=int(timeout), env=run_env
                )
                rc = int(rc_try)
                elapsed = float(elapsed_try)
                out = str(out_try)
                err = str(err_try)
                if rc == 0:
                    break
            text = f"{out}\n{err}"
            hG_max, hG_samples = _diag_max(text, "hG")
            hV_max, hV_samples = _diag_max(text, "hV")
            violW_max, violW_samples = _diag_max(text, "violW")
            lagV_max, lagV_samples = _diag_max(text, "lagV")
            asyncS_max, asyncS_samples = _diag_max(text, "asyncS")
            asyncB_max, asyncB_samples = _diag_max(text, "asyncB")
            if asyncS_samples == 0:
                asyncS_samples = lagV_samples
            if asyncB_samples == 0:
                asyncB_samples = lagV_samples
            send_pack_ms_max, send_pack_samples = _diag_max_float(text, "sendPackMs")
            send_wait_ms_max, send_wait_samples = _diag_max_float(text, "sendWaitMs")
            recv_poll_ms_max, recv_poll_samples = _diag_max_float(text, "recvPollMs")
            overlap_window_ms_max, overlap_window_samples = _diag_max_float(text, "overlapWinMs")
            if send_pack_samples == 0:
                send_pack_samples = lagV_samples
            if send_wait_samples == 0:
                send_wait_samples = lagV_samples
            if recv_poll_samples == 0:
                recv_poll_samples = lagV_samples
            if overlap_window_samples == 0:
                overlap_window_samples = lagV_samples
            wfgC_max, wfgC_samples = _diag_max(text, "wfgC")
            wfgO_max, wfgO_samples = _diag_max(text, "wfgO")
            wfgS_max, wfgS_samples = _diag_max(text, "wfgS")
            diag_samples = min(
                hG_samples,
                hV_samples,
                violW_samples,
                lagV_samples,
                asyncS_samples,
                asyncB_samples,
                send_pack_samples,
                send_wait_samples,
                recv_poll_samples,
                overlap_window_samples,
                wfgC_samples,
                wfgO_samples,
                wfgS_samples,
            )
            inv_ok = bool((hG_max == 0) and (hV_max == 0) and (violW_max == 0) and (lagV_max == 0))
            parse_ok = bool(diag_samples > 0)
            async_ok = bool(
                (overlap_i == 0)
                or (not require_async_evidence)
                or ((int(asyncS_max) > 0) and (int(asyncB_max) > 0))
            )
            overlap_win_ok = bool(
                (overlap_i == 0)
                or (not require_overlap_window)
                or (float(overlap_window_ms_max) > 0.0)
            )
            strict_ok = bool(
                ((not strict_invariants) or (inv_ok and parse_ok)) and async_ok and overlap_win_ok
            )
            rows.append(
                {
                    "overlap": overlap_i,
                    "cuda_aware_mpi": int(bool(enable_cuda_aware_mpi and overlap_i == 1)),
                    "returncode": int(rc),
                    "elapsed_sec": float(elapsed),
                    "hG_max": int(hG_max),
                    "hV_max": int(hV_max),
                    "violW_max": int(violW_max),
                    "lagV_max": int(lagV_max),
                    "async_send_msgs_max": int(asyncS_max),
                    "async_send_bytes_max": int(asyncB_max),
                    "async_evidence_ok": int(async_ok),
                    "send_pack_ms_max": float(send_pack_ms_max),
                    "send_wait_ms_max": float(send_wait_ms_max),
                    "recv_poll_ms_max": float(recv_poll_ms_max),
                    "overlap_window_ms_max": float(overlap_window_ms_max),
                    "overlap_window_ok": int(overlap_win_ok),
                    "diag_samples": int(diag_samples),
                    "wfgC_max": int(wfgC_max),
                    "wfgO_max": int(wfgO_max),
                    "wfgS_max": int(wfgS_max),
                    "wfgC_rate": float(wfgC_max) / float(wfgS_max) if int(wfgS_max) > 0 else 0.0,
                    "wfgC_per_100_steps": (
                        100.0 * float(wfgC_max) / float(n_steps) if int(n_steps) > 0 else 0.0
                    ),
                    "invariants_ok": int(inv_ok),
                    "strict_invariants_ok": int(strict_ok),
                    "attempts": int(attempts),
                    "simulated": 0,
                }
            )
        finally:
            try:
                os.unlink(cfg_tmp)
            except OSError:
                pass
    return rows


def _write_outputs(
    *, args, rows: list[dict[str, int | float]], cfg_path: Path, run_summary: dict
) -> None:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "overlap",
                "cuda_aware_mpi",
                "returncode",
                "elapsed_sec",
                "speedup_vs_blocking",
                "hG_max",
                "hV_max",
                "violW_max",
                "lagV_max",
                "async_send_msgs_max",
                "async_send_bytes_max",
                "async_evidence_ok",
                "send_pack_ms_max",
                "send_wait_ms_max",
                "recv_poll_ms_max",
                "overlap_window_ms_max",
                "overlap_window_ok",
                "diag_samples",
                "wfgC_max",
                "wfgO_max",
                "wfgS_max",
                "wfgC_rate",
                "wfgC_per_100_steps",
                "invariants_ok",
                "strict_invariants_ok",
                "attempts",
                "simulated",
            ]
        )
        t_base = next(
            (
                float(r["elapsed_sec"])
                for r in rows
                if int(r["overlap"]) == 0 and int(r["returncode"]) == 0
            ),
            None,
        )
        for r in rows:
            t = float(r["elapsed_sec"])
            sp = (
                (t_base / t)
                if (t_base is not None and t > 0 and int(r["returncode"]) == 0)
                else 0.0
            )
            w.writerow(
                [
                    int(r["overlap"]),
                    int(r["cuda_aware_mpi"]),
                    int(r["returncode"]),
                    f"{t:.6f}",
                    f"{sp:.6f}",
                    int(r["hG_max"]),
                    int(r["hV_max"]),
                    int(r["violW_max"]),
                    int(r["lagV_max"]),
                    int(r.get("async_send_msgs_max", 0)),
                    int(r.get("async_send_bytes_max", 0)),
                    int(r.get("async_evidence_ok", 1)),
                    f"{float(r.get('send_pack_ms_max', 0.0)):.6f}",
                    f"{float(r.get('send_wait_ms_max', 0.0)):.6f}",
                    f"{float(r.get('recv_poll_ms_max', 0.0)):.6f}",
                    f"{float(r.get('overlap_window_ms_max', 0.0)):.6f}",
                    int(r.get("overlap_window_ok", 1)),
                    int(r["diag_samples"]),
                    int(r.get("wfgC_max", 0)),
                    int(r.get("wfgO_max", 0)),
                    int(r.get("wfgS_max", 0)),
                    f"{float(r.get('wfgC_rate', 0.0)):.6f}",
                    f"{float(r.get('wfgC_per_100_steps', 0.0)):.6f}",
                    int(r["invariants_ok"]),
                    int(r["strict_invariants_ok"]),
                    int(r["attempts"]),
                    int(r.get("simulated", 0)),
                ]
            )

    with open(args.md, "w", encoding="utf-8") as f:
        f.write("# MPI Overlap Benchmark\n\n")
        f.write(f"- config: `{cfg_path}`\n")
        f.write(f"- ranks: `{int(run_summary['ranks'])}`\n")
        f.write(f"- cuda_aware_mode: `{bool(run_summary['cuda_aware'])}`\n")
        f.write(f"- strict_invariants: `{bool(run_summary['strict_invariants'])}`\n")
        f.write(f"- strict_zone_width: `{bool(run_summary['strict_zone_width'])}`\n")
        f.write(f"- simulated: `{bool(run_summary['simulated'])}`\n")
        if run_summary.get("profile_name"):
            f.write(f"- profile: `{run_summary['profile_name']}`\n")
        if run_summary.get("fabric"):
            f.write(f"- fabric: `{run_summary['fabric']}`\n")
        for r in rows:
            f.write(
                f"- overlap={int(r['overlap'])} cuda_aware={int(r['cuda_aware_mpi'])} "
                f"rc={int(r['returncode'])} elapsed={float(r['elapsed_sec']):.3f}s "
                f"hG={int(r['hG_max'])} hV={int(r['hV_max'])} violW={int(r['violW_max'])} lagV={int(r['lagV_max'])} "
                f"asyncS={int(r.get('async_send_msgs_max', 0))} asyncB={int(r.get('async_send_bytes_max', 0))} "
                f"async_ok={int(r.get('async_evidence_ok', 1))} "
                f"sendPackMs={float(r.get('send_pack_ms_max', 0.0)):.3f} sendWaitMs={float(r.get('send_wait_ms_max', 0.0)):.3f} recvPollMs={float(r.get('recv_poll_ms_max', 0.0)):.3f} overlapWinMs={float(r.get('overlap_window_ms_max', 0.0)):.3f} overlapWinOk={int(r.get('overlap_window_ok', 1))} "
                f"diag_samples={int(r['diag_samples'])} attempts={int(r['attempts'])} simulated={int(r.get('simulated', 0))}\n"
            )

    summary_path = args.summary or f"{args.out}.summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, **run_summary}, f, indent=2)


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark MPI overlap (blocking vs isend overlap)")
    p.add_argument("--profile", default="", help="Cluster profile YAML (optional)")
    p.add_argument("--config", default="", help="YAML config for TD-MPI run")
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--mpirun", default="")
    p.add_argument(
        "--overlap-list", default="", help="Comma-separated overlap modes (0=blocking,1=isend)"
    )
    p.add_argument(
        "--cuda-aware", action="store_true", help="Enable td.cuda_aware_mpi for overlap=1 runs"
    )
    p.add_argument("--timeout", type=int, default=0)
    p.add_argument("--out", default="results/mpi_overlap.csv")
    p.add_argument("--md", default="results/mpi_overlap.md")
    p.add_argument("--summary", default="", help="Optional summary JSON output path")
    p.add_argument("--retries", type=int, default=0, help="Attempts per overlap mode (>=1)")
    p.add_argument("--steps", type=int, default=0, help="Override run.n_steps in generated configs")
    p.add_argument(
        "--thermo-every", type=int, default=0, help="Override run.thermo_every in generated configs"
    )
    p.add_argument(
        "--fabric", default="", help="Free-text transport/fabric label for artifact normalization"
    )
    p.add_argument(
        "--simulate", action="store_true", help="Force simulated execution (no MPI launch)"
    )
    p.add_argument(
        "--no-strict-zone-width",
        action="store_true",
        help="Disable strict zone-width guardrail in generated benchmark configs",
    )
    p.add_argument(
        "--no-strict-invariants",
        action="store_true",
        help="Do not fail run when invariant counters are non-zero or missing",
    )
    p.add_argument(
        "--require-async-evidence",
        action="store_true",
        help="Require async overlap evidence (asyncS/asyncB > 0) for overlap=1 rows",
    )
    p.add_argument(
        "--require-overlap-window",
        action="store_true",
        help="Require overlap window evidence (overlapWinMs > 0) for overlap=1 rows",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Only validate inputs and print planned runs"
    )
    args = p.parse_args()

    profile = None
    if str(args.profile).strip():
        profile = load_cluster_profile(str(args.profile))

    cfg_rel = str(args.config).strip() or (
        str(profile.runtime.config)
        if profile is not None
        else "examples/td_1d_morse_static_rr_smoke4.yaml"
    )
    cfg_path = Path(cfg_rel)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    if not cfg_path.exists():
        print(f"[bench-mpi-overlap] config not found: {cfg_path}", file=sys.stderr)
        return 2

    overlap_text = str(args.overlap_list).strip() or (
        ",".join(str(x) for x in profile.runtime.overlap_list) if profile is not None else "0,1"
    )
    try:
        overlap_modes = _parse_overlap_list(overlap_text)
    except Exception as exc:
        print(f"[bench-mpi-overlap] invalid --overlap-list: {exc}", file=sys.stderr)
        return 2

    n_ranks = (
        int(args.n)
        if int(args.n) > 0
        else (int(profile.runtime.ranks_default[0]) if profile is not None else 4)
    )
    timeout = (
        int(args.timeout)
        if int(args.timeout) > 0
        else (int(profile.runtime.timeout_sec) if profile is not None else 180)
    )
    retries = (
        max(1, int(args.retries))
        if int(args.retries) > 0
        else (int(profile.runtime.retries) if profile is not None else 2)
    )
    n_steps = (
        int(args.steps)
        if int(args.steps) > 0
        else (int(profile.runtime.stability_steps) if profile is not None else 0)
    )
    thermo_every = (
        int(args.thermo_every)
        if int(args.thermo_every) > 0
        else (int(profile.runtime.stability_thermo_every) if profile is not None else 1)
    )

    strict_zone_width = bool(profile.runtime.strict_zone_width) if profile is not None else True
    if args.no_strict_zone_width:
        strict_zone_width = False
    strict_invariants = bool(profile.runtime.strict_invariants) if profile is not None else True
    if args.no_strict_invariants:
        strict_invariants = False
    require_async_evidence = bool(args.require_async_evidence)
    require_overlap_window = bool(args.require_overlap_window)

    prefer_simulated = bool(profile.runtime.prefer_simulated) if profile is not None else False
    allow_simulated = bool(profile.runtime.allow_simulated_cluster) if profile is not None else True
    simulate = bool(args.simulate or prefer_simulated)

    cuda_aware = bool(args.cuda_aware)
    if (not args.cuda_aware) and profile is not None:
        cuda_aware = bool(profile.scaling.cuda_aware_mpi)

    run_env = apply_profile_env(
        dict(os.environ), profile.runtime.env if profile is not None else {}
    )

    mpirun = _find_mpirun(
        str(args.mpirun).strip() or (profile.runtime.mpirun if profile is not None else "")
    )
    if not simulate and not mpirun:
        if allow_simulated:
            simulate = True
        else:
            print(
                "[bench-mpi-overlap] mpirun/mpiexec not found and simulated fallback is disabled",
                file=sys.stderr,
            )
            return 2

    if args.dry_run:
        print(f"[bench-mpi-overlap] dry-run config={cfg_path}")
        print(
            f"[bench-mpi-overlap] ranks={int(n_ranks)} overlap_modes={overlap_modes} cuda_aware={bool(cuda_aware)}"
        )
        print(
            f"[bench-mpi-overlap] mpirun={'<simulated>' if simulate else ('<auto>' if not args.mpirun else args.mpirun)}"
        )
        return 0

    with open(cfg_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    rows = _iter_rows(
        overlap_modes=overlap_modes,
        base=base,
        enable_cuda_aware_mpi=bool(cuda_aware),
        mpirun=str(mpirun or ""),
        n_ranks=int(n_ranks),
        timeout=int(timeout),
        root=ROOT,
        strict_invariants=bool(strict_invariants),
        require_async_evidence=bool(require_async_evidence),
        require_overlap_window=bool(require_overlap_window),
        retries=max(1, int(retries)),
        strict_zone_width=bool(strict_zone_width),
        n_steps=max(0, int(n_steps)),
        thermo_every=max(1, int(thermo_every)),
        simulate=bool(simulate),
        run_env=run_env,
    )

    env_keys = list((profile.runtime.env if profile is not None else {}).keys())
    run_summary = {
        "profile_name": str(profile.name) if profile is not None else "",
        "profile": profile.to_dict() if profile is not None else None,
        "profile_path": str(profile.source_path) if profile is not None else "",
        "config": str(cfg_path),
        "ranks": int(n_ranks),
        "timeout_sec": int(timeout),
        "retries": int(retries),
        "n_steps": int(n_steps),
        "thermo_every": int(thermo_every),
        "strict_invariants": bool(strict_invariants),
        "require_async_evidence": bool(require_async_evidence),
        "require_overlap_window": bool(require_overlap_window),
        "strict_zone_width": bool(strict_zone_width),
        "cuda_aware": bool(cuda_aware),
        "simulated": bool(simulate),
        "allow_simulated_cluster": bool(allow_simulated),
        "prefer_simulated": bool(prefer_simulated),
        "fabric": str(args.fabric).strip(),
        "mpirun": str(mpirun or ""),
        "env_snapshot": capture_env_snapshot(extra_keys=env_keys),
    }

    _write_outputs(args=args, rows=rows, cfg_path=cfg_path, run_summary=run_summary)

    print(f"[bench-mpi-overlap] wrote {args.out}")
    print(f"[bench-mpi-overlap] wrote {args.md}")
    summary_path = args.summary or f"{args.out}.summary.json"
    print(f"[bench-mpi-overlap] wrote {summary_path}")

    any_ok = any(int(r["returncode"]) == 0 for r in rows)
    all_ok = all(int(r["returncode"]) == 0 for r in rows)
    if strict_invariants:
        all_ok = bool(all_ok and all(int(r["strict_invariants_ok"]) == 1 for r in rows))
    if not any_ok:
        return 3
    return 0 if all_ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
