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

            idx = ffloat(r.get("idx")) if r.get("idx") is not None else float(i + 1)

            rows.append({
                "idx": idx,
                "score": ffloat(r.get("score_tail_mean_E")),
                "final_E": ffloat(r.get("final_E")),
                "arch": r.get("arch", "unknown"),
                "act": r.get("activation", "unknown"),
                "n_iter": r.get("n_iter", ""),
                "n_samples": r.get("n_samples", ""),
            })
    rows.sort(key=lambda x: x["idx"])
    return rows


if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing {CSV_PATH}. Run grid_search.py first.")

    rows = read_results(CSV_PATH)

    # score over index
    plt.figure()
    plt.plot([r["idx"] for r in rows], [r["score"] for r in rows], marker="o")
    plt.xlabel("Configuration index")
    plt.ylabel("Score (tail-mean energy)")
    plt.title("Score over grid configurations")
    out1 = os.path.join(SUMMARY_DIR, "score_over_index.png")
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close()

    # best score per arch
    best_arch = {}
    for r in rows:
        a = r["arch"]
        s = r["score"]
        if a not in best_arch or s < best_arch[a]:
            best_arch[a] = s
    archs = sorted(best_arch.keys())
    vals = [best_arch[a] for a in archs]
    plt.figure()
    plt.bar(archs, vals)
    plt.xlabel("Architecture")
    plt.ylabel("Best score (lower is better)")
    plt.title("Best score per architecture")
    out2 = os.path.join(SUMMARY_DIR, "best_score_per_arch.png")
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close()

    # best score per activation
    best_act = {}
    for r in rows:
        a = r["act"]
        s = r["score"]
        if a not in best_act or s < best_act[a]:
            best_act[a] = s
    acts = sorted(best_act.keys())
    vals = [best_act[a] for a in acts]
    plt.figure()
    plt.bar(acts, vals)
    plt.xlabel("Activation")
    plt.ylabel("Best score (lower is better)")
    plt.title("Best score per activation")
    out3 = os.path.join(SUMMARY_DIR, "best_score_per_activation.png")
    plt.savefig(out3, dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved plots:")
    print(" ", out1)
    print(" ", out2)
    print(" ", out3)
