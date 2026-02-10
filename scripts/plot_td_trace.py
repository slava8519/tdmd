from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_rows(paths: list[str]):
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                row["__path"] = p
                rows.append(row)
    return rows


def _state_code(s: str) -> int:
    s = (s or "").strip().upper()
    return {"F": 0, "D": 1, "P": 2, "W": 3, "S": 4}.get(s, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="td_trace.csv or directory/glob of trace files")
    ap.add_argument("--outdir", default="td_trace_plots")
    ap.add_argument("--rank", type=int, default=None, help="Filter to a single rank")
    args = ap.parse_args()

    if os.path.isdir(args.trace):
        paths = sorted(glob.glob(os.path.join(args.trace, "td_trace*.csv")))
    else:
        paths = sorted(glob.glob(args.trace))
        if not paths:
            paths = [args.trace]

    rows = _load_rows(paths)
    if args.rank is not None:
        rows = [r for r in rows if int(r.get("rank", 0)) == int(args.rank)]

    if not rows:
        raise SystemExit("no trace rows found")

    os.makedirs(args.outdir, exist_ok=True)

    # Per-rank event timeline (scatter)
    times = np.array([float(r["wall_time"]) for r in rows], dtype=float)
    ranks = np.array([int(r["rank"]) for r in rows], dtype=int)
    events = [r["event"] for r in rows]
    evt_types = sorted(set(events))
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(evt_types))))
    evt_color = {e: colors[i % len(colors)] for i, e in enumerate(evt_types)}
    plt.figure(figsize=(8, 3))
    for e in evt_types:
        idx = [i for i, ev in enumerate(events) if ev == e]
        if not idx:
            continue
        plt.scatter(times[idx], ranks[idx], s=6, color=evt_color[e], label=e, alpha=0.7)
    plt.xlabel("wall_time (s)")
    plt.ylabel("rank")
    plt.yticks(sorted(set(ranks)))
    plt.legend(loc="upper right", fontsize=6, ncol=2)
    plt.tight_layout()
    timeline_png = os.path.join(args.outdir, "timeline.png")
    plt.savefig(timeline_png)
    plt.close()

    # Heatmap state(zone, time) for a single rank (lowest if multiple)
    rank_sel = min(ranks) if args.rank is None else int(args.rank)
    r_rows = [r for r in rows if int(r["rank"]) == rank_sel]
    r_rows.sort(key=lambda r: float(r["wall_time"]))
    zones = sorted(set(int(r["zone_id"]) for r in r_rows))
    zone_index = {z: i for i, z in enumerate(zones)}
    n_events = len(r_rows)
    if zones and n_events:
        mat = np.zeros((len(zones), n_events), dtype=int)
        current = {z: 0 for z in zones}
        for i, r in enumerate(r_rows):
            zid = int(r["zone_id"])
            state = _state_code(r.get("state_after", ""))
            current[zid] = state
            for z, idx in zone_index.items():
                mat[idx, i] = current[z]
        plt.figure(figsize=(8, 4))
        plt.imshow(mat, aspect="auto", interpolation="nearest", origin="lower", cmap="viridis")
        plt.colorbar(label="state (F=0..S=4)")
        plt.yticks(range(len(zones)), zones)
        plt.xlabel("event index")
        plt.ylabel("zone_id")
        plt.title(f"state heatmap (rank {rank_sel})")
        plt.tight_layout()
        heat_png = os.path.join(args.outdir, "state_heatmap.png")
        plt.savefig(heat_png)
        plt.close()
    else:
        heat_png = None

    # Communication volume over time
    buckets = 50
    tmin, tmax = float(times.min()), float(times.max())
    if tmax <= tmin:
        tmax = tmin + 1e-6
    edges = np.linspace(tmin, tmax, buckets + 1)
    halo = np.zeros(buckets, dtype=float)
    mig = np.zeros(buckets, dtype=float)
    for r in rows:
        t = float(r["wall_time"])
        b = np.searchsorted(edges, t, side="right") - 1
        if b < 0 or b >= buckets:
            continue
        halo[b] += float(r.get("halo_ids_count", 0) or 0)
        mig[b] += float(r.get("migration_count", 0) or 0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    plt.figure(figsize=(8, 3))
    plt.plot(centers, halo, label="halo_ids_count")
    plt.plot(centers, mig, label="migration_count")
    plt.xlabel("wall_time (s)")
    plt.ylabel("count per bin")
    plt.legend()
    plt.tight_layout()
    comm_png = os.path.join(args.outdir, "comm_volume.png")
    plt.savefig(comm_png)
    plt.close()

    # HTML summary
    html_path = os.path.join(args.outdir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><body>\n")
        f.write("<h2>TD Trace Plots</h2>\n")
        f.write(f"<p>timeline: {os.path.basename(timeline_png)}</p>\n")
        f.write(f'<img src="{os.path.basename(timeline_png)}" width="800"/><br/>\n')
        if heat_png:
            f.write(f"<p>state heatmap: {os.path.basename(heat_png)}</p>\n")
            f.write(f'<img src="{os.path.basename(heat_png)}" width="800"/><br/>\n')
        f.write(f"<p>comm volume: {os.path.basename(comm_png)}</p>\n")
        f.write(f'<img src="{os.path.basename(comm_png)}" width="800"/><br/>\n')
        f.write("</body></html>\n")

    print(f"[plot_td_trace] wrote {args.outdir}")


if __name__ == "__main__":
    main()
