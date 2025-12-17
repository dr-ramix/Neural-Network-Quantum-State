import os
import csv
from itertools import product
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from config import Config, ITERS, SAMPLES, LRS, ARCHS, ACTS, SHIFTS
from run import run_vmc, print_devices


# -------------------------
# Grid size (sanity)
# -------------------------
TOTAL = len(ITERS) * len(SAMPLES) * len(LRS) * len(ARCHS) * len(ACTS) * len(SHIFTS)


# -------------------------
# Output folders
# -------------------------
OUTDIR = "outputs"
CSV_PATH = os.path.join(OUTDIR, "results.csv")
HISTDIR = os.path.join(OUTDIR, "histories")
PLOTDIR = os.path.join(OUTDIR, "plots")
TRACE_DIR = os.path.join(PLOTDIR, "energy_traces")
SUMMARY_DIR = os.path.join(PLOTDIR, "summary")

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(HISTDIR, exist_ok=True)
os.makedirs(TRACE_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)


# -------------------------
# CSV fields
# -------------------------
CSV_FIELDS = [
    "idx",
    "score",
    "final_E",
    "final_Eerr",
    "per_site_E",
    "per_site_Eerr",
    "min_E",
    "n_params",
    "n_sites",
    "it", "ns", "lr", "shift", "arch", "act",
    "history_file",
]


def append_csv(path: str, row: Dict[str, Any]):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


# -------------------------
# ASCII table printer
# -------------------------
COLS = [
    "idx", "score", "final_E", "Eerr", "E/site", "params",
    "it", "ns", "lr", "shift", "arch", "act"
]
W = {
    "idx": 4,
    "score": 12,
    "final_E": 12,
    "Eerr": 10,
    "E/site": 12,
    "params": 8,
    "it": 4,
    "ns": 6,
    "lr": 10,
    "shift": 8,
    "arch": 10,
    "act": 9,
}


def _fmt(x, width: int, prec: int = 4) -> str:
    if x is None:
        s = "-"
    elif isinstance(x, int):
        s = str(x)
    elif isinstance(x, float):
        ax = abs(x)
        if ax != 0 and (ax < 1e-3 or ax >= 1e4):
            s = f"{x:.{prec}e}"
        else:
            s = f"{x:.{prec}f}"
    else:
        s = str(x)

    if len(s) > width:
        s = s[: width - 1] + "…"
    return s.rjust(width)


def print_header():
    header = " | ".join(c.center(W[c]) for c in COLS)
    sep = "-+-".join("-" * W[c] for c in COLS)
    print(header)
    print(sep)


def print_row(r: Dict[str, Any]):
    line = " | ".join(_fmt(r.get(c), W[c]) for c in COLS)
    print(line)


# -------------------------
# Plot helpers
# -------------------------
def save_trace_plot(iters, energy, err, title, outpath):
    plt.figure()
    plt.plot(iters, energy)
    plt.fill_between(iters, energy - err, energy + err, alpha=0.2)
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title(title)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def make_summary_plots(rows):
    # score over index
    x = [r["idx"] for r in rows]
    y = [r["score"] for r in rows]

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Configuration index")
    plt.ylabel("Score (tail-mean energy)")
    plt.title("Score over grid configurations")
    plt.savefig(os.path.join(SUMMARY_DIR, "score_over_index.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # best score per architecture
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
    plt.savefig(os.path.join(SUMMARY_DIR, "best_score_per_arch.png"), dpi=200, bbox_inches="tight")
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
    plt.savefig(os.path.join(SUMMARY_DIR, "best_score_per_activation.png"), dpi=200, bbox_inches="tight")
    plt.close()


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    print_devices()
    print(f"\nGrid size: {TOTAL} runs")
    print(f"CSV:      {CSV_PATH}")
    print(f"History:  {HISTDIR}")
    print(f"Plots:    {PLOTDIR}\n")

    print_header()

    idx = 0
    best = None
    rows_for_summary = []

    for it, ns, lr, shift, arch, act in product(ITERS, SAMPLES, LRS, SHIFTS, ARCHS, ACTS):
        idx += 1

        cfg = Config(
            arch=arch,
            activation=act,
            n_iter=it,
            n_samples=ns,
            learning_rate=lr,
            diag_shift=shift,
        )

        res = run_vmc(cfg)

        run_id = f"idx{idx:03d}_it{it}_ns{ns}_lr{lr:.0e}_sh{shift:.0e}_arch{arch}_act{act}"
        hist_path = os.path.join(HISTDIR, f"{run_id}.npz")

        np.savez_compressed(
            hist_path,
            iters=res["iters"],
            energy=res["energy"],
            err=res["err"],
            score=res["score"],
            final_E=res["final_E"],
            final_Eerr=res["final_Eerr"],
            per_site_E=res["per_site_E"],
            per_site_Eerr=res["per_site_Eerr"],
            min_E=res["min_E"],
            n_params=res["n_params"],
            n_sites=res["n_sites"],
            it=it, ns=ns, lr=lr, shift=shift, arch=arch, act=act,
        )

        # per-run plot
        trace_title = f"{arch} | {act} | it={it} ns={ns} lr={lr:.0e} shift={shift:.0e}"
        trace_png = os.path.join(TRACE_DIR, f"{run_id}.png")
        save_trace_plot(res["iters"], res["energy"], res["err"], trace_title, trace_png)

        row_csv = {
            "idx": idx,
            "score": res["score"],
            "final_E": res["final_E"],
            "final_Eerr": res["final_Eerr"],
            "per_site_E": res["per_site_E"],
            "per_site_Eerr": res["per_site_Eerr"],
            "min_E": res["min_E"],
            "n_params": res["n_params"],
            "n_sites": res["n_sites"],
            "it": it, "ns": ns, "lr": lr, "shift": shift, "arch": arch, "act": act,
            "history_file": hist_path,
        }
        append_csv(CSV_PATH, row_csv)
        rows_for_summary.append(row_csv)

        # table row
        row_table = {
            "idx": idx,
            "score": res["score"],
            "final_E": res["final_E"],
            "Eerr": res["final_Eerr"],
            "E/site": res["per_site_E"],
            "params": res["n_params"],
            "it": it,
            "ns": ns,
            "lr": lr,
            "shift": shift,
            "arch": arch,
            "act": act,
        }
        print_row(row_table)

        # update best
        if best is None or res["score"] < best["score"]:
            best = {"score": res["score"], "row": row_csv}

    # summary plots
    make_summary_plots(rows_for_summary)

    print("\nDONE")
    print(f"Saved CSV:   {CSV_PATH}")
    print(f"Saved NPZ:   {HISTDIR}")
    print(f"Saved plots: {PLOTDIR}")

    if best is not None:
        b = best["row"]
        print("\nBEST (by score = tail-mean energy)")
        print(f"  score={b['score']:.10f}")
        print(f"  arch={b['arch']} act={b['act']} it={b['it']} ns={b['ns']} lr={b['lr']:.0e} shift={b['shift']:.0e}")
        print(f"  final_E={b['final_E']:.10f} ± {b['final_Eerr']:.10f}")
