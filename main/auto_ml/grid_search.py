import os
import csv
import json
from itertools import product
from typing import Dict, Any, Tuple

from config import Config
from run import run_vmc, print_devices


# -------------------------
# YOUR FINAL TUNING SPACE
# -------------------------
ITERS   = [400, 800]
SAMPLES = [1024, 2048]
LRS     = [1e-3, 3e-3]
ARCHS   = ["N", "N_N", "N_N_N"]
ACTS    = ["log_cosh", "silu", "gelu"]
SHIFTS  = [1e-2]

TOTAL = len(ITERS) * len(SAMPLES) * len(LRS) * len(ARCHS) * len(ACTS) * len(SHIFTS)


# -------------------------
# OUTPUT FOLDERS
# -------------------------
OUTDIR = "outputs"
HISTDIR = os.path.join(OUTDIR, "histories")
PLOTDIR = os.path.join(OUTDIR, "plots")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(HISTDIR, exist_ok=True)
os.makedirs(PLOTDIR, exist_ok=True)

CSV_PATH = os.path.join(OUTDIR, "results.csv")
META_PATH = os.path.join(OUTDIR, "grid_meta.json")


# -------------------------
# Pretty table (ASCII)
# -------------------------
COLS = ["idx", "score", "final_E", "Eerr", "it", "ns", "lr", "shift", "arch", "act", "params"]
W = {"idx": 5, "score": 12, "final_E": 12, "Eerr": 10, "it": 4, "ns": 6, "lr": 10, "shift": 8, "arch": 8, "act": 9, "params": 8}

def _fmt(x, width, prec=4):
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

def print_row(row: Dict[str, Any]):
    line = " | ".join(_fmt(row.get(c), W[c]) for c in COLS)
    print(line)


# -------------------------
# Resume support
# -------------------------
def _key(it: int, ns: int, lr: float, shift: float, arch: str, act: str) -> Tuple[int, int, float, float, str, str]:
    return (it, ns, float(lr), float(shift), arch, act)

def load_done_keys(csv_path: str):
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            done.add(_key(
                int(r["it"]), int(r["ns"]), float(r["lr"]), float(r["shift"]), r["arch"], r["act"]
            ))
    return done

def append_csv(csv_path: str, fieldnames, rowdict):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(rowdict)


if __name__ == "__main__":
    print_devices()
    print(f"\nGrid size: {TOTAL} runs")
    print(f"Writing CSV to: {CSV_PATH}")
    print(f"Writing histories to: {HISTDIR}\n")

    # save meta once (useful for your paper)
    if not os.path.exists(META_PATH):
        with open(META_PATH, "w") as f:
            json.dump({
                "ITERS": ITERS,
                "SAMPLES": SAMPLES,
                "LRS": LRS,
                "ARCHS": ARCHS,
                "ACTS": ACTS,
                "SHIFTS": SHIFTS,
                "TOTAL": TOTAL,
            }, f, indent=2)

    done = load_done_keys(CSV_PATH)
    if done:
        print(f"Resuming: {len(done)} already done, {TOTAL - len(done)} remaining.\n")

    csv_fields = [
        "idx",
        "score",
        "final_E",
        "final_Eerr",
        "per_site_E",
        "per_site_Eerr",
        "min_E",
        "n_params",
        "n_sites",
        "rhat_last",
        "taucorr_last",
        "it", "ns", "lr", "shift", "arch", "act",
        "history_file",
    ]

    best = None
    idx = 0

    print_header()

    for it, ns, lr, shift, arch, act in product(ITERS, SAMPLES, LRS, SHIFTS, ARCHS, ACTS):
        idx += 1
        key = _key(it, ns, lr, shift, arch, act)
        if key in done:
            continue

        cfg = Config(
            arch=arch,
            activation=act,
            n_iter=it,
            n_samples=ns,
            learning_rate=lr,
            diag_shift=shift,
        )

        res = run_vmc(cfg)

        # Save history (for plots)
        run_id = f"it{it}_ns{ns}_lr{lr:.0e}_sh{shift:.0e}_arch{arch}_act{act}"
        hist_path = os.path.join(HISTDIR, f"{run_id}.npz")
        # store arrays + minimal metadata
        import numpy as np
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
            n_params=res["n_params"],
            n_sites=res["n_sites"],
            rhat_last=res["rhat_last"] if res["rhat_last"] is not None else np.nan,
            taucorr_last=res["taucorr_last"] if res["taucorr_last"] is not None else np.nan,
            it=it, ns=ns, lr=lr, shift=shift,
            arch=arch, act=act,
        )

        # Print a clean row
        row = {
            "idx": idx,
            "score": res["score"],
            "final_E": res["final_E"],
            "Eerr": res["final_Eerr"],
            "it": it,
            "ns": ns,
            "lr": lr,
            "shift": shift,
            "arch": arch,
            "act": act,
            "params": res["n_params"],
        }
        print_row(row)

        # Update best
        if best is None or res["score"] < best["score"]:
            best = {"score": res["score"], "cfg": cfg, "res": res, "hist": hist_path}

        # Append CSV
        append_csv(CSV_PATH, csv_fields, {
            "idx": idx,
            "score": res["score"],
            "final_E": res["final_E"],
            "final_Eerr": res["final_Eerr"],
            "per_site_E": res["per_site_E"],
            "per_site_Eerr": res["per_site_Eerr"],
            "min_E": res["min_E"],
            "n_params": res["n_params"],
            "n_sites": res["n_sites"],
            "rhat_last": res["rhat_last"],
            "taucorr_last": res["taucorr_last"],
            "it": it, "ns": ns, "lr": lr, "shift": shift, "arch": arch, "act": act,
            "history_file": hist_path,
        })

    if best is not None:
        print("\n" + "-" * 92)
        print("BEST FOUND (by tail-mean score)")
        b = best["cfg"]
        print(f"score={best['score']:.10f}")
        print(f"arch={b.arch} act={b.activation} it={b.n_iter} ns={b.n_samples} lr={b.learning_rate:.3e} shift={b.diag_shift:.3e}")
        print(f"final_E={best['res']['final_E']:.10f} ± {best['res']['final_Eerr']:.10f}")
        print(f"history={best['hist']}")
        print("-" * 92)
