from __future__ import annotations
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def _read_csv(path: str):
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = [row for row in r]
    return rows

def plot_metrics_csv(csv_path: str, out_dir: str):
    rows = _read_csv(csv_path)
    if not rows:
        return
    os.makedirs(out_dir, exist_ok=True)
    metrics = [
        "max_dr","max_dv","max_dE","max_dT","max_dP",
        "final_dr","final_dv","final_dE","final_dT","final_dP",
        "rms_dr","rms_dv","rms_dE","rms_dT","rms_dP",
    ]
    # group by case
    cases = sorted(set(r["case"] for r in rows))
    for m in metrics:
        plt.figure()
        for case in cases:
            y = [float(r[m]) for r in rows if r["case"] == case]
            x = np.arange(len(y))
            plt.plot(x, y, label=case)
        plt.yscale("log")
        plt.xlabel("sweep index")
        plt.ylabel(m)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{m}.png"))
        plt.close()
