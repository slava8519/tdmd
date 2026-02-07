from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
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


def _run_case(*, profile_path: str, ranks: int, overlap: int, cuda_aware: bool,
              steps: int, thermo_every: int, timeout: int, retries: int,
              simulate: bool, out_dir: Path) -> dict[str, Any]:
    out_csv = out_dir / f"stability_n{int(ranks)}_m{int(overlap)}.csv"
    out_md = out_dir / f"stability_n{int(ranks)}_m{int(overlap)}.md"
    out_json = out_dir / f"stability_n{int(ranks)}_m{int(overlap)}.summary.json"

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
        "--steps",
        str(max(1, int(steps))),
        "--thermo-every",
        str(max(1, int(thermo_every))),
    ]
    if bool(cuda_aware):
        cmd.append("--cuda-aware")
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Cluster long-run stability benchmark and strict gate")
    ap.add_argument("--profile", default="examples/cluster/cluster_profile_smoke.yaml")
    ap.add_argument("--out", default="results/cluster_stability.csv")
    ap.add_argument("--md", default="results/cluster_stability.md")
    ap.add_argument("--json", default="results/cluster_stability_summary.json")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--simulate", action="store_true", help="Force simulated mode")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    profile = load_cluster_profile(str(args.profile))

    if args.dry_run:
        overlaps = sorted(set([0, int(profile.stability.overlap)]))
        print(f"[cluster-stability] dry-run profile={profile.source_path}")
        print(f"[cluster-stability] ranks={profile.stability.ranks} overlaps={overlaps}")
        print(f"[cluster-stability] steps={profile.runtime.stability_steps} thermo_every={profile.runtime.stability_thermo_every}")
        return 0

    out_csv = Path(args.out)
    out_md = Path(args.md)
    out_json = Path(args.json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="tdmd_cluster_stability_") as td:
        tmp_dir = Path(td)
        overlaps = sorted(set([0, int(profile.stability.overlap)]))
        for rk in list(profile.stability.ranks):
            rk_i = int(rk)
            for ov in overlaps:
                use_cuda_aware = bool(profile.stability.cuda_aware_mpi and int(ov) == 1)
                run = _run_case(
                    profile_path=profile.source_path,
                    ranks=rk_i,
                    overlap=int(ov),
                    cuda_aware=use_cuda_aware,
                    steps=int(profile.runtime.stability_steps),
                    thermo_every=int(profile.runtime.stability_thermo_every),
                    timeout=int(profile.runtime.timeout_sec),
                    retries=int(profile.runtime.retries),
                    simulate=bool(args.simulate or profile.runtime.prefer_simulated),
                    out_dir=tmp_dir,
                )
                reasons: list[str] = list(run.get("reasons", []))
                gate_ok = bool(int(run["returncode"]) == 0 and int(run["strict_invariants_ok"]) == 1)
                if int(run["hG_max"]) > int(profile.stability.max_hG):
                    gate_ok = False
                    reasons.append(f"hG>{int(profile.stability.max_hG)}")
                if int(run["hV_max"]) > int(profile.stability.max_hV):
                    gate_ok = False
                    reasons.append(f"hV>{int(profile.stability.max_hV)}")
                if int(run["violW_max"]) > int(profile.stability.max_violW):
                    gate_ok = False
                    reasons.append(f"violW>{int(profile.stability.max_violW)}")
                if int(run["lagV_max"]) > int(profile.stability.max_lagV):
                    gate_ok = False
                    reasons.append(f"lagV>{int(profile.stability.max_lagV)}")
                if int(run["diag_samples"]) < int(profile.stability.min_diag_samples):
                    gate_ok = False
                    reasons.append(f"diag_samples<{int(profile.stability.min_diag_samples)}")
                rows.append(
                    {
                        **run,
                        "ranks": int(rk_i),
                        "overlap": int(ov),
                        "cuda_aware_mpi": int(use_cuda_aware),
                        "steps": int(profile.runtime.stability_steps),
                        "thermo_every": int(profile.runtime.stability_thermo_every),
                        "ok": bool(gate_ok),
                        "reasons": reasons,
                    }
                )

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
            "min_diag_samples": min((int(r["diag_samples"]) for r in rows), default=0),
        },
        "rows": rows,
        "env_snapshot": capture_env_snapshot(extra_keys=list(profile.runtime.env.keys())),
    }

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "ranks",
                "overlap",
                "cuda_aware_mpi",
                "steps",
                "thermo_every",
                "elapsed_sec",
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
                    int(r["overlap"]),
                    int(r["cuda_aware_mpi"]),
                    int(r["steps"]),
                    int(r["thermo_every"]),
                    f"{float(r['elapsed_sec']):.6f}",
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
        f.write("# Cluster Stability Benchmark\n\n")
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
                f"- ranks={int(r['ranks'])} overlap={int(r['overlap'])} cuda_aware={int(r['cuda_aware_mpi'])} "
                f"elapsed={float(r['elapsed_sec']):.3f}s hG={int(r['hG_max'])} hV={int(r['hV_max'])} "
                f"violW={int(r['violW_max'])} lagV={int(r['lagV_max'])} diag={int(r['diag_samples'])} "
                f"ok={bool(r['ok'])} simulated={int(r.get('simulated', 0))}\n"
            )

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[cluster-stability] wrote {out_csv}")
    print(f"[cluster-stability] wrote {out_md}")
    print(f"[cluster-stability] wrote {out_json}")

    if args.strict and not bool(summary["ok_all"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
