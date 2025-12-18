import os
import csv
import json
from itertools import product
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from config import Config, ITERS, SAMPLES, LRS, ARCHS, ACTS, SHIFTS
from run import run_vmc

import socket
import jax
print("Host:", socket.gethostname())
print("JAX backend:", jax.default_backend())
print("JAX devices:", jax.devices())

# -------------------------
# Output paths
# -------------------------
OUTDIR = "outputs"
CSV_PATH = os.path.join(OUTDIR, "results.csv")
TXT_TABLE_PATH = os.path.join(OUTDIR, "table.txt")
HISTDIR = os.path.join(OUTDIR, "histories")
PLOTDIR = os.path.join(OUTDIR, "plots")
TRACE_DIR = os.path.join(PLOTDIR, "energy_traces")
SUMMARY_DIR = os.path.join(PLOTDIR, "summary")

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(HISTDIR, exist_ok=True)
os.makedirs(TRACE_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)


TOTAL = len(ITERS) * len(SAMPLES) * len(LRS) * len(ARCHS) * len(ACTS) * len(SHIFTS)


# -------------------------
# CSV schema (titles)
# -------------------------
CSV_FIELDS = [
    "idx",
    "score_tail_mean_E",
    "final_E",
    "final_Eerr",
    "per_site_E",
    "per_site_Eerr",
    "min_E",
    "n_params",
    "n_sites",
    "rhat_last",
    "taucorr_last",
    "n_iter", "n_samples", "learning_rate", "diag_shift", "arch", "activation",
    "history_npz",
    "history_json",
]


def write_csv_row(row: Dict[str, Any]):
    exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


# -------------------------
# Pretty table (console + txt)
# -------------------------
COLS = [
    "idx", "score", "final_E", "Eerr", "E/site", "params",
    "it", "ns", "lr", "shift", "arch", "act", "finite?"
]
W = {
    "idx": 4, "score": 12, "final_E": 12, "Eerr": 10, "E/site": 12, "params": 8,
    "it": 4, "ns": 7, "lr": 10, "shift": 8, "arch": 10, "act": 9, "finite?": 8,
}


def _fmt(x, width: int, prec: int = 4) -> str:
    if x is None:
        s = "-"
    elif isinstance(x, bool):
        s = "yes" if x else "no"
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


def table_header() -> str:
    header = " | ".join(c.center(W[c]) for c in COLS)
    sep = "-+-".join("-" * W[c] for c in COLS)
    return header + "\n" + sep


def table_row(r: Dict[str, Any]) -> str:
    return " | ".join(_fmt(r.get(c), W[c]) for c in COLS)


def write_table_line(line: str):
    with open(TXT_TABLE_PATH, "a") as f:
        f.write(line + "\n")


# -------------------------
# Robust plotting
# -------------------------
def save_trace_plot(iters, energy, err, title, outpath):
    iters = np.asarray(iters)
    energy = np.asarray(energy)
    err = np.asarray(err)

    mask = np.isfinite(energy) & np.isfinite(err)
    it = iters[mask]
    e = energy[mask]
    s = err[mask]

    plt.figure()

    if len(e) == 0:
        plt.title(title + " (no finite points)")
        plt.xlabel("Iteration")
        plt.ylabel("Energy")
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close()
        return

    # robust y-limits: percentiles to avoid huge spikes flattening the plot
    lo, hi = np.percentile(e, [2, 98])
    pad = 0.1 * (hi - lo) if hi > lo else (abs(hi) + 1.0)

    plt.plot(it, e)
    plt.fill_between(it, e - s, e + s, alpha=0.2)
    plt.ylim(lo - pad, hi + pad)

    # annotate if clipping likely occurred
    if np.max(np.abs(e)) > max(abs(lo), abs(hi)) * 50:
        plt.title(title + " (outliers clipped)")
    else:
        plt.title(title)

    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def make_summary_plots(rows):
    # score vs index
    x = [r["idx"] for r in rows]
    y = [r["score_tail_mean_E"] for r in rows]
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("Configuration index")
    plt.ylabel("Score (tail-mean energy)")
    plt.title("Score over grid configurations")
    plt.savefig(os.path.join(SUMMARY_DIR, "score_over_index.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # best score per arch
    best_arch = {}
    for r in rows:
        a = r["arch"]
        s = r["score_tail_mean_E"]
        if a not in best_arch or s < best_arch[a]:
            best_arch[a] = s
    archs = sorted(best_arch.keys())
    vals = [best_arch[a] for a in archs]
    plt.figure()
    plt.bar(archs, vals)
    plt.xlabel("Architecture")
    plt.ylabel("Best score (lower is better)")
    plt.title("Best score per architecture")
    plt.savefig(os.path.join(SUMMARY_DIR, "best_score_per_arch.png"), dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # fresh table file
    if os.path.exists(TXT_TABLE_PATH):
        os.remove(TXT_TABLE_PATH)

    print(f"\nGrid size: {TOTAL} runs")
    print(f"CSV:   {CSV_PATH}")
    print(f"TXT:   {TXT_TABLE_PATH}")
    print(f"NPZ:   {HISTDIR}")
    print(f"Plots: {PLOTDIR}\n")

    header = table_header()
    print(header)
    write_table_line(header)

    rows_for_summary = []
    best = None
    idx = 0

    for it, ns, lr, shift, arch, act in product(ITERS, SAMPLES, LRS, SHIFTS, ARCHS, ACTS):
        idx += 1

        cfg = Config(
            arch=arch,
            activation=act,
            n_iter=int(it),
            n_samples=int(ns),
            learning_rate=float(lr),
            diag_shift=float(shift),
        )

        res = run_vmc(cfg)

        # finite check (helps catch divergence)
        finite_ok = bool(np.all(np.isfinite(res["energy_mean"])) and np.all(np.isfinite(res["energy_sigma"])))

        run_id = f"idx{idx:03d}_it{it}_ns{ns}_lr{lr:.0e}_sh{shift:.0e}_arch{arch}_act{act}"

        # Save per-run config + metadata as JSON
        json_path = os.path.join(HISTDIR, f"{run_id}.json")
        with open(json_path, "w") as f:
            json.dump({
                "idx": idx,
                "config": {
                    "L": cfg.L, "J1": cfg.J1, "J2": cfg.J2, "pbc": cfg.pbc, "total_sz": cfg.total_sz,
                    "arch": cfg.arch, "activation": cfg.activation, "use_output_bias": cfg.use_output_bias,
                    "param_dtype": str(cfg.param_dtype),
                    "d_max": cfg.d_max, "n_chains": cfg.n_chains, "n_samples": cfg.n_samples,
                    "n_discard_per_chain": cfg.n_discard_per_chain,
                    "learning_rate": cfg.learning_rate, "diag_shift": cfg.diag_shift,
                    "n_iter": cfg.n_iter, "seed": cfg.seed,
                }
            }, f, indent=2)

        # Save full optimization path + all histories into NPZ
        npz_path = os.path.join(HISTDIR, f"{run_id}.npz")

        # flatten “all_histories” safely into NPZ (store per-history arrays)
        npz_payload = {
            "iters": np.asarray(res["iters"]),
            "energy_mean": np.asarray(res["energy_mean"]),
            "energy_sigma": np.asarray(res["energy_sigma"]),
            "score": float(res["score"]),
            "final_E": float(res["final_E"]),
            "final_Eerr": float(res["final_Eerr"]),
            "per_site_E": float(res["per_site_E"]),
            "per_site_Eerr": float(res["per_site_Eerr"]),
            "min_E": float(res["min_E"]),
            "n_sites": int(res["n_sites"]),
            "n_params": int(res["n_params"]),
            "rhat_last": np.nan if res["rhat_last"] is None else float(res["rhat_last"]),
            "taucorr_last": np.nan if res["taucorr_last"] is None else float(res["taucorr_last"]),
            "n_iter": int(it),
            "n_samples": int(ns),
            "learning_rate": float(lr),
            "diag_shift": float(shift),
        }

        if res["energy_var"] is not None:
            npz_payload["energy_var"] = np.asarray(res["energy_var"])

        # add all histories (if present)
        # keys like: Energy_Mean, Energy_Sigma, ...
        for hname, hdict in res["all_histories"].items():
            # store only if arrays exist
            if hdict.get("iters") is not None:
                npz_payload[f"{hname}__iters"] = np.asarray(hdict["iters"])
            for field in ["Mean", "Sigma", "Variance", "R_hat", "TauCorr"]:
                arr = hdict.get(field)
                if arr is not None:
                    npz_payload[f"{hname}__{field}"] = np.asarray(arr)

        np.savez_compressed(npz_path, **npz_payload)

        # Per-run plot (robust)
        title = f"{arch} | {act} | it={it} ns={ns} lr={lr:.0e} shift={shift:.0e}"
        trace_png = os.path.join(TRACE_DIR, f"{run_id}.png")
        save_trace_plot(res["iters"], res["energy_mean"], res["energy_sigma"], title, trace_png)

        # Row for table
        row_tbl = {
            "idx": idx,
            "score": float(res["score"]),
            "final_E": float(res["final_E"]),
            "Eerr": float(res["final_Eerr"]),
            "E/site": float(res["per_site_E"]),
            "params": int(res["n_params"]),
            "it": int(it),
            "ns": int(ns),
            "lr": float(lr),
            "shift": float(shift),
            "arch": arch,
            "act": act,
            "finite?": finite_ok,
        }
        line = table_row(row_tbl)
        print(line)
        write_table_line(line)

        # CSV row
        row_csv = {
            "idx": idx,
            "score_tail_mean_E": float(res["score"]),
            "final_E": float(res["final_E"]),
            "final_Eerr": float(res["final_Eerr"]),
            "per_site_E": float(res["per_site_E"]),
            "per_site_Eerr": float(res["per_site_Eerr"]),
            "min_E": float(res["min_E"]),
            "n_params": int(res["n_params"]),
            "n_sites": int(res["n_sites"]),
            "rhat_last": res["rhat_last"],
            "taucorr_last": res["taucorr_last"],
            "n_iter": int(it),
            "n_samples": int(ns),
            "learning_rate": float(lr),
            "diag_shift": float(shift),
            "arch": arch,
            "activation": act,
            "history_npz": npz_path,
            "history_json": json_path,
        }
        write_csv_row(row_csv)
        rows_for_summary.append(row_csv)

        if best is None or float(res["score"]) < float(best["score"]):
            best = {"score": float(res["score"]), "row": row_csv}

    make_summary_plots(rows_for_summary)

    print("\nDONE")
    print(f"Saved CSV: {CSV_PATH}")
    print(f"Saved TXT: {TXT_TABLE_PATH}")
    print(f"Saved NPZ: {HISTDIR}")
    print(f"Saved plots: {PLOTDIR}")

    if best is not None:
        b = best["row"]
        print("\nBEST (by score = tail-mean energy)")
        print(f"  score={b['score_tail_mean_E']:.10f}")
        print(f"  arch={b['arch']} act={b['activation']} it={b['n_iter']} ns={b['n_samples']} lr={b['learning_rate']:.0e} shift={b['diag_shift']:.0e}")
        print(f"  final_E={b['final_E']:.10f} ± {b['final_Eerr']:.10f}")
