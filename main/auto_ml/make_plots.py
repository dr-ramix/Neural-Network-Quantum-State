import os
import csv
import math
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt


OUTDIR = "outputs"
CSV_PATH = os.path.join(OUTDIR, "results.csv")
PLOTDIR = os.path.join(OUTDIR, "plots")
SUMMARY_DIR = os.path.join(PLOTDIR, "summary")
os.makedirs(SUMMARY_DIR, exist_ok=True)


def read_results(csv_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            def ffloat(x):
                try:
                    return float(x)
                except Exception:
                    return math.nan
            def fint(x):
                try:
                    return int(float(x))
                except Exception:
                    return -1

            rows.append({
                "idx": fint(r["idx"]),
                "score": ffloat(r["score"]),
                "final_E": ffloat(r["final_E"]),
                "final_Eerr": ffloat(r["final_Eerr"]),
                "arch": r["arch"],
                "act": r["act"],
            })
    rows.sort(key=lambda x: x["idx"])
    return rows


def plot_score_over_index(rows):
    x = [r["idx"] for r in rows]
    y = [r["score"] for r in rows]
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Configuration index")
    plt.ylabel("Score (tail-mean energy)")
    plt.title("Score over grid configurations")
    out = os.path.join(SUMMARY_DIR, "score_over_index.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


def plot_best_score_per_arch(rows):
    best = {}
    for r in rows:
        a = r["arch"]
        s = r["score"]
        if a not in best or s < best[a]:
            best[a] = s

    archs = sorted(best.keys())
    vals = [best[a] for a in archs]

    plt.figure()
    plt.bar(archs, vals)
    plt.xlabel("Architecture")
    plt.ylabel("Best score (lower is better)")
    plt.title("Best score per architecture")
    out = os.path.join(SUMMARY_DIR, "best_score_per_arch.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


def plot_best_score_per_activation(rows):
    best = {}
    for r in rows:
        a = r["act"]
        s = r["score"]
        if a not in best or s < best[a]:
            best[a] = s

    acts = sorted(best.keys())
    vals = [best[a] for a in acts]

    plt.figure()
    plt.bar(acts, vals)
    plt.xlabel("Activation")
    plt.ylabel("Best score (lower is better)")
    plt.title("Best score per activation")
    out = os.path.join(SUMMARY_DIR, "best_score_per_activation.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing {CSV_PATH}. Run grid_search.py first.")

    rows = read_results(CSV_PATH)

    out1 = plot_score_over_index(rows)
    out2 = plot_best_score_per_arch(rows)
    out3 = plot_best_score_per_activation(rows)

    print("Saved summary plots:")
    print(" ", out1)
    print(" ", out2)
    print(" ", out3)
