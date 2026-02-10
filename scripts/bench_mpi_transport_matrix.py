from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tdmd.cluster_profile import capture_env_snapshot, load_cluster_profile


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


def _run_case(
    *,
    profile_path: str,
    ranks: int,
    overlap: int,
    cuda_aware: bool,
    fabric: str,
    timeout: int,
    retries: int,
    simulate: bool,
    out_dir: Path,
    extra_env: dict[str, str],
) -> dict[str, Any]:
    out_csv = out_dir / f"transport_n{int(ranks)}_m{int(overlap)}_{fabric}.csv"
    out_md = out_dir / f"transport_n{int(ranks)}_m{int(overlap)}_{fabric}.md"
    out_json = out_dir / f"transport_n{int(ranks)}_m{int(overlap)}_{fabric}.summary.json"

    cmd = [
        sys.executable,
        "scripts/bench_mpi_overlap.py",
        "--profile",
        str(profile_path),
        "--n",
        str(int(ranks)),
        "--overlap-list",
        str(int(overlap)),
        "--fabric",
        str(fabric),
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
    if bool(simulate):
        cmd.append("--simulate")

    run_env = dict(os.environ)
    for k, v in extra_env.items():
        run_env[str(k)] = str(v)

    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, env=run_env)
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


def main() -> int:
    ap = argparse.ArgumentParser(description="MPI transport matrix benchmark/gate (fabric-aware)")
    ap.add_argument("--profile", default="examples/cluster/cluster_profile_smoke.yaml")
    ap.add_argument("--out", default="results/mpi_transport_matrix.csv")
    ap.add_argument("--md", default="results/mpi_transport_matrix.md")
    ap.add_argument("--json", default="results/mpi_transport_matrix_summary.json")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--simulate", action="store_true", help="Force simulated mode")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    profile = load_cluster_profile(str(args.profile))

    if args.dry_run:
        print(f"[transport-matrix] dry-run profile={profile.source_path}")
        print(f"[transport-matrix] ranks={profile.transport_matrix.ranks}")
        for e in profile.transport_matrix.entries:
            print(
                f"  - {e.name}: overlap={e.overlap} cuda_aware={e.cuda_aware_mpi} fabric={e.fabric}"
            )
        return 0

    out_csv = Path(args.out)
    out_md = Path(args.md)
    out_json = Path(args.json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="tdmd_transport_matrix_") as td:
        tmp_dir = Path(td)
        for rk in list(profile.transport_matrix.ranks):
            rk_i = int(rk)
            for entry in list(profile.transport_matrix.entries):
                run = _run_case(
                    profile_path=profile.source_path,
                    ranks=rk_i,
                    overlap=int(entry.overlap),
                    cuda_aware=bool(entry.cuda_aware_mpi),
                    fabric=str(entry.fabric),
                    timeout=int(profile.runtime.timeout_sec),
                    retries=int(profile.runtime.retries),
                    simulate=bool(args.simulate or profile.runtime.prefer_simulated),
                    out_dir=tmp_dir,
                    extra_env=dict(entry.env),
                )
                rows.append(
                    {
                        **run,
                        "ranks": rk_i,
                        "profile_name": str(entry.name),
                        "overlap": int(entry.overlap),
                        "cuda_aware_mpi": int(bool(entry.cuda_aware_mpi)),
                        "fabric": str(entry.fabric),
                        "entry_env": dict(entry.env),
                    }
                )

    # compute per-rank blocking baseline for communication-cost comparison
    base_elapsed_by_rank: dict[int, float] = {}
    for r in rows:
        if int(r["overlap"]) == 0 and int(r["returncode"]) == 0:
            base_elapsed_by_rank.setdefault(int(r["ranks"]), float(r["elapsed_sec"]))

    for r in rows:
        rk = int(r["ranks"])
        base = float(base_elapsed_by_rank.get(rk, 0.0))
        elapsed = float(r["elapsed_sec"])
        speedup = (base / elapsed) if base > 0 and elapsed > 0 else 0.0
        reasons: list[str] = list(r.get("reasons", []))
        ok = bool(int(r["returncode"]) == 0 and int(r["strict_invariants_ok"]) == 1)
        if int(r["overlap"]) == 1 and speedup <= 0.0:
            ok = False
            reasons.append("speedup_vs_blocking<=0")
        r["speedup_vs_blocking"] = float(speedup)
        r["ok"] = bool(ok)
        r["reasons"] = reasons

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
            "min_speedup_vs_blocking": min(
                (float(r["speedup_vs_blocking"]) for r in rows), default=0.0
            ),
        },
        "rows": rows,
        "env_snapshot": capture_env_snapshot(extra_keys=list(profile.runtime.env.keys())),
    }

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "ranks",
                "profile_name",
                "fabric",
                "overlap",
                "cuda_aware_mpi",
                "elapsed_sec",
                "speedup_vs_blocking",
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
                    int(r["ranks"]),
                    str(r["profile_name"]),
                    str(r["fabric"]),
                    int(r["overlap"]),
                    int(r["cuda_aware_mpi"]),
                    f"{float(r['elapsed_sec']):.6f}",
                    f"{float(r['speedup_vs_blocking']):.6f}",
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
        f.write("# MPI Transport Matrix\n\n")
        f.write(f"- profile: `{profile.source_path}`\n")
        f.write(f"- strict: `{bool(args.strict)}`\n")
        f.write(f"- total: `{total}`\n")
        f.write(f"- ok: `{ok_n}`\n")
        f.write(f"- fail: `{fail_n}`\n")
        f.write(f"- ok_all: `{bool(summary['ok_all'])}`\n")
        f.write("\n## Worst\n\n")
        f.write(json.dumps(summary["worst"], indent=2))
        f.write("\n\n## Rows\n\n")
        for r in rows:
            f.write(
                f"- ranks={int(r['ranks'])} profile={r['profile_name']} fabric={r['fabric']} "
                f"overlap={int(r['overlap'])} cuda_aware={int(r['cuda_aware_mpi'])} "
                f"elapsed={float(r['elapsed_sec']):.3f}s speedup={float(r['speedup_vs_blocking']):.3f} "
                f"hG={int(r['hG_max'])} hV={int(r['hV_max'])} violW={int(r['violW_max'])} lagV={int(r['lagV_max'])} "
                f"ok={bool(r['ok'])} simulated={int(r.get('simulated', 0))}\n"
            )

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[transport-matrix] wrote {out_csv}")
    print(f"[transport-matrix] wrote {out_md}")
    print(f"[transport-matrix] wrote {out_json}")

    if args.strict and not bool(summary["ok_all"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
