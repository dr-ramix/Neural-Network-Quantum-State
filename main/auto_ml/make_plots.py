import os
import csv
import math
from typing import List, Dict, Any

import matplotlib.pyplot as plt


OUTDIR = "outputs"
CSV_PATH = os.path.join(OUTDIR, "results.csv")
SUMMARY_DIR = os.path.join(OUTDIR, "plots", "summary")
os.makedirs(SUMMARY_DIR, exist_ok=True)


def read_results(csv_path: str) -> List[Dict[str, Any]]:
    rows = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            def ffloat(x):
                try:
                    return float(x)
                except Exception:
                    return math.nan

            # Use idx column if it exists, otherwise fallback to row number
            idx = ffloat(r["idx"]) if "idx" in r else float(i + 1)

            rows.append({
                "idx": idx,
                "score": ffloat(r.get("score")),
                "final_E": ffloat(r.get("final_E")),
                "final_Eerr": ffloat(r.get("final_Eerr")),
                "arch": r.get("arch", "unknown"),
                "act": r.get("act", "unknown"),
            })

    return rows


def plot_score_over_index(rows):
    x = [r["idx"] for r in rows]
    y = [r["score"] for r in rows]

    plt.figure()
    plt.plot(x, y, marker="o")
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
        a, s = r["arch"], r["score"]
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
        a, s = r["act"], r["score"]
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

    print("Saved plots:")
    print(" ", out1)
    print(" ", out2)
    print(" ", out3)
