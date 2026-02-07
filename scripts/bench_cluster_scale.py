from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tdmd.cluster_profile import capture_env_snapshot, load_cluster_profile


def _load_envelope(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in list(data.get("rows", [])):
        if not isinstance(row, dict):
            continue
        mode = str(row.get("mode", "")).strip().lower()
        ranks = int(row.get("ranks", 0))
        if mode and ranks > 0:
            out[(mode, ranks)] = dict(row)
    return out


def _read_overlap_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
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


def _pick_row(rows: list[dict[str, str]], overlap: int) -> dict[str, str] | None:
    for r in rows:
        if _as_int(r, "overlap", default=-1) == int(overlap):
            return r
    return None


def _run_case(*, profile_path: str, ranks: int, overlap: int, cuda_aware: bool,
              steps: int, timeout: int, retries: int, simulate: bool, out_dir: Path) -> dict[str, Any]:
    out_csv = out_dir / f"scale_overlap_n{int(ranks)}_m{int(overlap)}.csv"
    out_md = out_dir / f"scale_overlap_n{int(ranks)}_m{int(overlap)}.md"
    out_json = out_dir / f"scale_overlap_n{int(ranks)}_m{int(overlap)}.summary.json"

    cmd = [
        sys.executable,
        "scripts/bench_mpi_overlap.py",
        "--profile",
        str(profile_path),
        "--n",
        str(int(ranks)),
        "--overlap-list",
        str(int(overlap)),
        "--out",
        str(out_csv),
        "--md",
        str(out_md),
        "--summary",
        str(out_json),
        "--timeout",
        str(int(timeout)),
        "--retries",
        str(max(1, int(retries))),
    ]
    if bool(cuda_aware):
        cmd.append("--cuda-aware")
    if int(steps) > 0:
        cmd.extend(["--steps", str(int(steps))])
    if bool(simulate):
        cmd.append("--simulate")

    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    rows = _read_overlap_csv(out_csv)
    pick = _pick_row(rows, overlap=overlap)
    if pick is None:
        return {
            "returncode": int(proc.returncode),
            "elapsed_sec": 0.0,
            "hG_max": 0,
            "hV_max": 0,
            "violW_max": 0,
            "lagV_max": 0,
            "diag_samples": 0,
            "strict_invariants_ok": 0,
            "simulated": int(bool(simulate)),
            "reasons": ["missing_overlap_row"],
            "out_csv": str(out_csv),
            "out_md": str(out_md),
            "out_json": str(out_json),
            "stdout": str(proc.stdout or ""),
            "stderr": str(proc.stderr or ""),
        }

    return {
        "returncode": int(_as_int(pick, "returncode", default=proc.returncode)),
        "elapsed_sec": float(_as_float(pick, "elapsed_sec", default=0.0)),
        "hG_max": int(_as_int(pick, "hG_max", default=0)),
        "hV_max": int(_as_int(pick, "hV_max", default=0)),
        "violW_max": int(_as_int(pick, "violW_max", default=0)),
        "lagV_max": int(_as_int(pick, "lagV_max", default=0)),
        "diag_samples": int(_as_int(pick, "diag_samples", default=0)),
        "strict_invariants_ok": int(_as_int(pick, "strict_invariants_ok", default=0)),
        "simulated": int(_as_int(pick, "simulated", default=int(bool(simulate)))),
        "reasons": [],
        "out_csv": str(out_csv),
        "out_md": str(out_md),
        "out_json": str(out_json),
        "stdout": str(proc.stdout or ""),
        "stderr": str(proc.stderr or ""),
    }


def _node_count(ranks: int, ranks_per_node: int) -> int:
    return int(math.ceil(float(max(1, int(ranks))) / float(max(1, int(ranks_per_node)))))


def main() -> int:
    ap = argparse.ArgumentParser(description="Cluster strong/weak scaling benchmark and strict gate")
    ap.add_argument("--profile", default="examples/cluster/cluster_profile_smoke.yaml")
    ap.add_argument("--out", default="results/cluster_scale.csv")
    ap.add_argument("--md", default="results/cluster_scale.md")
    ap.add_argument("--json", default="results/cluster_scale_summary.json")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--simulate", action="store_true", help="Force simulated mode")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    profile = load_cluster_profile(str(args.profile))

    if args.dry_run:
        print(f"[cluster-scale] dry-run profile={profile.source_path}")
        print(f"[cluster-scale] strong_ranks={profile.scaling.strong_ranks} weak_ranks={profile.scaling.weak_ranks}")
        print(f"[cluster-scale] overlap={profile.scaling.overlap} cuda_aware={profile.scaling.cuda_aware_mpi}")
        return 0

    out_csv = Path(args.out)
    out_md = Path(args.md)
    out_json = Path(args.json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tdmd_cluster_scale_") as td:
        tmp_dir = Path(td)
        rows: list[dict[str, Any]] = []
        for mode in ("strong", "weak"):
            ranks_list = list(profile.scaling.strong_ranks if mode == "strong" else profile.scaling.weak_ranks)
            base_rank = int(min(ranks_list))
            mode_results: list[dict[str, Any]] = []
            for rk in ranks_list:
                rk_i = int(rk)
                steps = int(profile.runtime.stability_steps)
                if mode == "weak":
                    steps = max(1, int(round(float(profile.runtime.stability_steps) * (float(rk_i) / float(base_rank)))))
                run = _run_case(
                    profile_path=profile.source_path,
                    ranks=rk_i,
                    overlap=int(profile.scaling.overlap),
                    cuda_aware=bool(profile.scaling.cuda_aware_mpi),
                    steps=int(steps),
                    timeout=int(profile.runtime.timeout_sec),
                    retries=int(profile.runtime.retries),
                    simulate=bool(args.simulate or profile.runtime.prefer_simulated),
                    out_dir=tmp_dir,
                )
                run["mode"] = mode
                run["ranks"] = rk_i
                run["base_rank"] = base_rank
                run["steps"] = int(steps)
                run["node_count"] = _node_count(rk_i, int(profile.runtime.ranks_per_node))
                mode_results.append(run)

            base_elapsed = 0.0
            for rr in mode_results:
                if int(rr["ranks"]) == base_rank and int(rr["returncode"]) == 0:
                    base_elapsed = float(rr["elapsed_sec"])
                    break
            if base_elapsed <= 0.0:
                base_elapsed = 1.0

            for rr in mode_results:
                elapsed = float(rr["elapsed_sec"])
                rank_ratio = float(max(1, int(rr["ranks"]))) / float(base_rank)
                speedup = (base_elapsed / elapsed) if elapsed > 0 else 0.0
                efficiency = (speedup / rank_ratio) if rank_ratio > 0 else 0.0
                weak_elapsed_ratio = (elapsed / base_elapsed) if base_elapsed > 0 else 0.0
                gate_ok = bool(int(rr["returncode"]) == 0 and int(rr["strict_invariants_ok"]) == 1)
                reasons: list[str] = list(rr.get("reasons", []))
                if mode == "strong" and int(rr["ranks"]) != base_rank:
                    if speedup < float(profile.scaling.min_speedup):
                        gate_ok = False
                        reasons.append(
                            f"strong_speedup<{profile.scaling.min_speedup:.3f} ({speedup:.3f})"
                        )
                    if efficiency < float(profile.scaling.min_efficiency):
                        gate_ok = False
                        reasons.append(
                            f"strong_efficiency<{profile.scaling.min_efficiency:.3f} ({efficiency:.3f})"
                        )
                if mode == "weak" and int(rr["ranks"]) != base_rank:
                    if weak_elapsed_ratio > float(profile.scaling.max_weak_elapsed_ratio):
                        gate_ok = False
                        reasons.append(
                            f"weak_elapsed_ratio>{profile.scaling.max_weak_elapsed_ratio:.3f} ({weak_elapsed_ratio:.3f})"
                        )
                rows.append(
                    {
                        **rr,
                        "speedup": float(speedup),
                        "efficiency": float(efficiency),
                        "weak_elapsed_ratio": float(weak_elapsed_ratio),
                        "ok": bool(gate_ok),
                        "reasons": reasons,
                    }
                )

    envelope_file = str(profile.metadata.get("scale_envelope_file", "")).strip()
    envelope_path = Path(envelope_file) if envelope_file else None
    envelope_map: dict[tuple[str, int], dict[str, Any]] = {}
    if envelope_path is not None and envelope_file:
        if not envelope_path.is_absolute():
            envelope_path = ROOT / envelope_path
        envelope_map = _load_envelope(envelope_path)
        for r in rows:
            key = (str(r["mode"]).lower(), int(r["ranks"]))
            spec = envelope_map.get(key)
            if spec is None:
                r["ok"] = False
                rs = list(r.get("reasons", []))
                rs.append("missing_envelope_row")
                r["reasons"] = rs
                continue
            if str(r["mode"]).lower() == "strong":
                min_speedup = float(spec.get("min_speedup", profile.scaling.min_speedup))
                min_eff = float(spec.get("min_efficiency", profile.scaling.min_efficiency))
                if float(r["speedup"]) < min_speedup:
                    r["ok"] = False
                    rs = list(r.get("reasons", []))
                    rs.append(f"envelope_speedup<{min_speedup:.3f}")
                    r["reasons"] = rs
                if float(r["efficiency"]) < min_eff:
                    r["ok"] = False
                    rs = list(r.get("reasons", []))
                    rs.append(f"envelope_efficiency<{min_eff:.3f}")
                    r["reasons"] = rs
            else:
                max_ratio = float(spec.get("max_weak_elapsed_ratio", profile.scaling.max_weak_elapsed_ratio))
                if float(r["weak_elapsed_ratio"]) > max_ratio:
                    r["ok"] = False
                    rs = list(r.get("reasons", []))
                    rs.append(f"envelope_weak_ratio>{max_ratio:.3f}")
                    r["reasons"] = rs

    total = int(len(rows))
    ok_n = int(sum(1 for r in rows if bool(r.get("ok", False))))
    fail_n = int(total - ok_n)
    summary = {
        "profile": profile.to_dict(),
        "total": total,
        "ok": ok_n,
        "fail": fail_n,
        "ok_all": bool(ok_n == total),
        "worst": {
            "max_hG": max((int(r["hG_max"]) for r in rows), default=0),
            "max_hV": max((int(r["hV_max"]) for r in rows), default=0),
            "max_violW": max((int(r["violW_max"]) for r in rows), default=0),
            "max_lagV": max((int(r["lagV_max"]) for r in rows), default=0),
            "min_speedup": min((float(r["speedup"]) for r in rows), default=0.0),
            "min_efficiency": min((float(r["efficiency"]) for r in rows), default=0.0),
            "max_weak_elapsed_ratio": max((float(r["weak_elapsed_ratio"]) for r in rows), default=0.0),
        },
        "rows": rows,
        "env_snapshot": capture_env_snapshot(extra_keys=list(profile.runtime.env.keys())),
        "envelope_file": str(envelope_path) if envelope_path is not None else "",
        "envelope_rows": int(len(envelope_map)),
    }

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "mode",
                "ranks",
                "node_count",
                "steps",
                "elapsed_sec",
                "speedup",
                "efficiency",
                "weak_elapsed_ratio",
                "hG_max",
                "hV_max",
                "violW_max",
                "lagV_max",
                "diag_samples",
                "strict_invariants_ok",
                "returncode",
                "simulated",
                "ok",
                "reasons",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    str(r["mode"]),
                    int(r["ranks"]),
                    int(r["node_count"]),
                    int(r["steps"]),
                    f"{float(r['elapsed_sec']):.6f}",
                    f"{float(r['speedup']):.6f}",
                    f"{float(r['efficiency']):.6f}",
                    f"{float(r['weak_elapsed_ratio']):.6f}",
                    int(r["hG_max"]),
                    int(r["hV_max"]),
                    int(r["violW_max"]),
                    int(r["lagV_max"]),
                    int(r["diag_samples"]),
                    int(r["strict_invariants_ok"]),
                    int(r["returncode"]),
                    int(r.get("simulated", 0)),
                    int(bool(r["ok"])),
                    ";".join(str(x) for x in list(r.get("reasons", []))),
                ]
            )

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Cluster Scale Benchmark\n\n")
        f.write(f"- profile: `{profile.source_path}`\n")
        f.write(f"- strict: `{bool(args.strict)}`\n")
        f.write(f"- total: `{total}`\n")
        f.write(f"- ok: `{ok_n}`\n")
        f.write(f"- fail: `{fail_n}`\n")
        f.write(f"- ok_all: `{bool(summary['ok_all'])}`\n")
        if summary.get("envelope_file"):
            f.write(f"- envelope_file: `{summary['envelope_file']}`\n")
            f.write(f"- envelope_rows: `{int(summary.get('envelope_rows', 0))}`\n")
        f.write("\n## Worst\n\n")
        f.write(json.dumps(summary["worst"], indent=2))
        f.write("\n\n## Rows\n\n")
        for r in rows:
            f.write(
                f"- {r['mode']} ranks={int(r['ranks'])} nodes={int(r['node_count'])} "
                f"elapsed={float(r['elapsed_sec']):.3f}s speedup={float(r['speedup']):.3f} "
                f"eff={float(r['efficiency']):.3f} weak_ratio={float(r['weak_elapsed_ratio']):.3f} "
                f"ok={bool(r['ok'])} simulated={int(r.get('simulated', 0))}\n"
            )

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[cluster-scale] wrote {out_csv}")
    print(f"[cluster-scale] wrote {out_md}")
    print(f"[cluster-scale] wrote {out_json}")

    if args.strict and not bool(summary["ok_all"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
