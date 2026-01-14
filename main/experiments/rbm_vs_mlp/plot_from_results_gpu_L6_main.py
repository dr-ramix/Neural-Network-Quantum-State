#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# HARD-CODED LITERATURE VALUES (energy per site)
# ============================================================
PAPER_TRUE_E_SITE: Dict[float, float] = {
    0.4: -0.52975,
    0.5: -0.50381,
    0.6: -0.49518,
    1.0: -0.71436,
}


# ============================================================
# Matplotlib style (paper-ready)
# ============================================================
def set_paper_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,

        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",

        "axes.spines.top": False,
        "axes.spines.right": False,

        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.fancybox": True,

        "lines.linewidth": 2.0,
    })


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load NetKet history CSV (your format)
# ============================================================
def load_history_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    CSV format:
      # comments...
      iter,energy_mean,energy_sigma,energy_per_site_mean,energy_per_site_sigma
      ...
    """
    rows: List[List[str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            if line.lower().startswith("iter,"):
                continue
            parts = line.strip().split(",")
            if len(parts) == 5:
                rows.append(parts)

    if not rows:
        raise RuntimeError(f"No numeric data rows in {csv_path}")

    arr = np.asarray(rows, dtype=float)
    iters = arr[:, 0].astype(int)
    e_site = arr[:, 3]
    e_site_err = arr[:, 4]
    return iters, e_site, e_site_err


# ============================================================
# Load run_meta.json
# ============================================================
def load_run_meta(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "run_meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def make_subtitle(meta: Dict[str, Any], J2: float) -> str:
    """
    Clean subtitle: physics + training params; no backend/gpu/device text.
    """
    parts: List[str] = []

    L = meta.get("L", None)
    N = meta.get("n_sites", None)
    J1 = meta.get("J1", None)

    if L is not None and N is not None:
        parts.append(f"L={int(L)} (N={int(N)})")
    if J1 is not None:
        parts.append(f"J1={float(J1):g}")
    parts.append(f"J2={J2:.2f}")

    ns = meta.get("n_samples", None)
    if ns is not None:
        parts.append(f"samples={int(ns)}")

    ds = meta.get("diag_shift", None)
    if ds is not None:
        parts.append(f"diag_shift={float(ds):g}")

    opt = meta.get("optimizer", {})
    if isinstance(opt, dict) and "learning_rate" in opt:
        parts.append(f"lr={float(opt['learning_rate']):g}")

    return " | ".join(parts)


# ============================================================
# Plot helper (BIGGER + no overlap + minimal legend indicating errors)
# ============================================================
def plot_energy(
    outbase: Path,
    main_title: str,
    subtitle: str,
    iters: np.ndarray,
    curves: List[Tuple[str, np.ndarray, np.ndarray]],
    J2: float,
) -> None:
    set_paper_style()

    # Bigger figure for paper readability
    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    # Reserve vertical space at the top for title + subtitle (prevents overlap)
    fig.subplots_adjust(top=0.76)

    iters = np.asarray(iters)

    for label, y, yerr in curves:
        y = np.asarray(y)
        yerr = np.asarray(yerr)

        # Line first, then reuse its color for error bars/band
        (line,) = ax.plot(
            iters, y,
            marker="o",
            markersize=2.5,
            label=label + r": $E/N \pm \sigma/N$",
            zorder=3
        )
        c = line.get_color()

        # Error bars for EVERY point, same color as line
        ax.errorbar(
            iters, y, yerr=yerr,
            fmt="none",
            ecolor=c,
            elinewidth=1.1,
            capsize=2.5,
            alpha=0.95,
            zorder=2
        )

        # Subtle uncertainty band (same color)
        ax.fill_between(
            iters, y - yerr, y + yerr,
            color=c, alpha=0.10, zorder=1
        )

    # Literature reference (black dashed)
    if J2 in PAPER_TRUE_E_SITE:
        ax.axhline(
            PAPER_TRUE_E_SITE[J2],
            linestyle="--",
            linewidth=2.4,
            color="black",
            label="Literature value",
            zorder=0
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Energy per site $E/N$")

    # Title + subtitle with clear separation (subtitle outside axes)
    fig.suptitle(main_title, y=0.96, fontsize=16)
    fig.text(0.5, 0.88, subtitle, ha="center", va="center", fontsize=12)

    ax.minorticks_on()
    ax.tick_params(which="both", direction="out")

    ax.legend(
        loc="best",
        handlelength=2.8,
        borderpad=0.6,
        labelspacing=0.4
    )

    ensure_dir(outbase.parent)
    fig.savefig(outbase.with_suffix(".png"))
    fig.savefig(outbase.with_suffix(".pdf"))
    plt.close(fig)


# ============================================================
# Scan result tree
# ============================================================
CSV_RE = re.compile(r"^(MLP|RBM)_(real|complex)_J2_([0-9]+\.[0-9]+)_history\.csv$")


def scan_runs(root: Path) -> Dict[float, Dict[Tuple[str, str], Dict[str, Any]]]:
    runs: Dict[float, Dict[Tuple[str, str], Dict[str, Any]]] = {}

    for csv in root.rglob("*_history.csv"):
        m = CSV_RE.match(csv.name)
        if not m:
            continue

        arch, dtype, j2s = m.groups()
        J2 = float(j2s)

        it, e, err = load_history_csv(csv)
        meta = load_run_meta(csv.parent)

        runs.setdefault(J2, {})[(arch, dtype)] = {
            "iters": it,
            "e": e,
            "err": err,
            "meta": meta,
            "dir": csv.parent,
        }

    return runs


# ============================================================
# Main
# ============================================================
def main() -> None:
    # Run this script from: nqs/main/experiments/rbm_vs_mlp
    root = Path("results_gpu_L6_main")
    if not root.exists():
        raise SystemExit(
            "results_gpu_L6_main not found. Run this from rbm_vs_mlp/ "
            "or change 'root' in the script."
        )

    runs = scan_runs(root)
    if not runs:
        raise SystemExit("No matching *_history.csv files found.")

    # ---- Per-run plots ----
    for J2, d in sorted(runs.items()):
        for (arch, dtype), r in d.items():
            subtitle = make_subtitle(r["meta"], J2)

            plot_energy(
                outbase=r["dir"] / f"{arch}_{dtype}_Esite_clean",
                main_title="VMC Energy per Site Convergence",
                subtitle=subtitle,
                iters=r["iters"],
                curves=[(f"{arch} ({dtype})", r["e"], r["err"])],
                J2=J2,
            )

    # ---- Per-J2 combined plots ----
    comp_root = root / "compare_per_j2"
    ensure_dir(comp_root)

    for J2, d in sorted(runs.items()):
        min_len = min(len(v["iters"]) for v in d.values())

        iters_common: Optional[np.ndarray] = None
        curves: List[Tuple[str, np.ndarray, np.ndarray]] = []

        for arch in ("MLP", "RBM"):
            for dtype in ("real", "complex"):
                key = (arch, dtype)
                if key not in d:
                    continue
                r = d[key]
                iters_common = r["iters"][:min_len] if iters_common is None else iters_common
                curves.append((f"{arch} ({dtype})", r["e"][:min_len], r["err"][:min_len]))

        if iters_common is None or len(curves) < 2:
            continue

        any_meta = next(iter(d.values()))["meta"]
        subtitle = make_subtitle(any_meta, J2)

        outdir = comp_root / f"J2_{J2:.2f}"
        ensure_dir(outdir)

        plot_energy(
            outbase=outdir / f"compare_all_J2_{J2:.2f}_clean",
            main_title="VMC Energy per Site Convergence (comparison)",
            subtitle=subtitle,
            iters=iters_common,
            curves=curves,
            J2=J2,
        )

    print("✓ Done. Clean plots written as PNG + PDF.")
    print("  - Per-run: next to each run folder as *_Esite_clean.*")
    print("  - Combined: compare_per_j2/J2_xx/compare_all_*_clean.*")


if __name__ == "__main__":
    main()
