import os
import csv
import math
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt


OUTDIR = "outputs"
CSV_PATH = os.path.join(OUTDIR, "results.csv")
HISTDIR = os.path.join(OUTDIR, "histories")
PLOTDIR = os.path.join(OUTDIR, "plots")
TRACE_DIR = os.path.join(PLOTDIR, "energy_traces_topk")
SUMMARY_DIR = os.path.join(PLOTDIR, "summary")
os.makedirs(TRACE_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)


def read_results(csv_path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # parse numeric fields safely
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
                "it": fint(r["it"]),
                "ns": fint(r["ns"]),
                "lr": ffloat(r["lr"]),
                "shift": ffloat(r["shift"]),
                "arch": r["arch"],
                "act": r["act"],
                "n_params": fint(r["n_params"]),
                "history_file": r["history_file"],
            })
    rows.sort(key=lambda x: x["idx"])
    return rows


def plot_score_over_index(rows: List[Dict[str, Any]]):
    idx = [r["idx"] for r in rows]
    score = [r["score"] for r in rows]

    plt.figure()
    plt.plot(idx, score)
    plt.xlabel("Configuration index")
    plt.ylabel("Score (tail-mean energy)")
    plt.title("Grid search score over configurations")
    out = os.path.join(SUMMARY_DIR, "score_over_index.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return out


def plot_best_score_per_arch(rows: List[Dict[str, Any]]):
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


def plot_topk_traces(rows: List[Dict[str, Any]], k: int = 6):
    rows_sorted = sorted(rows, key=lambda r: r["score"])
    top = rows_sorted[:k]

    outs = []
    for r in top:
        path = r["history_file"]
        if not path or not os.path.exists(path):
            continue
        data = np.load(path)
        iters = data["iters"]
        energy = data["energy"]
        err = data["err"]

        plt.figure()
        plt.plot(iters, energy, label="Energy")
        # error band
        plt.fill_between(iters, energy - err, energy + err, alpha=0.2)

        title = f"arch={r['arch']} act={r['act']} it={r['it']} ns={r['ns']} lr={r['lr']:.0e}"
        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Energy")

        fname = f"trace_idx{r['idx']}_arch{r['arch']}_act{r['act']}_it{r['it']}_ns{r['ns']}_lr{r['lr']:.0e}.png"
        out = os.path.join(TRACE_DIR, fname)
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        outs.append(out)

    return outs


if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing {CSV_PATH}. Run grid_search.py first.")

    rows = read_results(CSV_PATH)

    out1 = plot_score_over_index(rows)
    out2 = plot_best_score_per_arch(rows)
    trace_outs = plot_topk_traces(rows, k=6)

    print("Saved plots:")
    print(" ", out1)
    print(" ", out2)
    for p in trace_outs:
        print(" ", p)
