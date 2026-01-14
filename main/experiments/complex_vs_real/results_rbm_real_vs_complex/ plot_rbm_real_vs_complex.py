#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot-only: RBM REAL vs COMPLEX per J2 (from existing CSVs).

Expected structure (run from results_rbm_real_vs_complex/):
  RBM_REAL/J2_0.40/rbm_real_J2_0.40_history.csv
  RBM_COMPLEX/J2_0.40/rbm_complex_J2_0.40_history.csv
  ... similarly for J2_0.50, J2_0.60, J2_1.00

Creates per-J2 plots (REAL vs COMPLEX):
  - error bars for every iteration (same color as line)
  - subtle error band
  - literature value line (black dashed)
  - two sizes (small, big)
  - PNG + PDF

No NetKet/JAX required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Hard-coded literature values (energy per site)
# -----------------------------
LITERATURE_E_SITE: Dict[float, float] = {
    0.4: -0.52975,
    0.5: -0.50381,
    0.6: -0.49518,
    1.0: -0.71436,
}

# J2 values you want to plot
J2_LIST = [0.4, 0.5, 0.6, 1.0]

FIG_SMALL = (7.6, 4.6)
FIG_BIG   = (10.8, 6.4)


def set_paper_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        "font.size": 12,
        "axes.titlesize": 15,
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


def load_history_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Reads CSV format:
      # comments...
      iter,energy_mean,energy_sigma,energy_per_site_mean,energy_per_site_sigma
      ...
    Returns: iters, e_site, e_site_err, meta
    """
    iters: List[int] = []
    e_site: List[float] = []
    e_err: List[float] = []
    meta: dict = {}

    with csv_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            # Parse metadata from comment lines (optional)
            if s.startswith("#"):
                if "L=" in s and "J2=" in s:
                    # "# RBM history | dtype=COMPLEX | L=6 (N_sites=36) | J1=1.0 | J2=0.4"
                    try:
                        meta["L"] = int(s.split("L=")[1].split()[0])
                    except Exception:
                        pass
                    try:
                        meta["N_sites"] = int(s.split("N_sites=")[1].split(")")[0])
                    except Exception:
                        pass
                    try:
                        meta["J1"] = float(s.split("J1=")[1].split()[0])
                    except Exception:
                        pass
                    try:
                        meta["J2"] = float(s.split("J2=")[1].split()[0])
                    except Exception:
                        pass

                if "n_samples=" in s:
                    # "# Sampler=... | n_samples=10000 | discard=50 | seed=1234"
                    try:
                        meta["n_samples"] = int(s.split("n_samples=")[1].split()[0])
                    except Exception:
                        pass

                if "diag_shift=" in s:
                    # "# Optimizer=... | ... | SR diag_shift=0.01"
                    try:
                        meta["diag_shift"] = float(s.split("diag_shift=")[1].split()[0])
                    except Exception:
                        pass
                continue

            if s.lower().startswith("iter,"):
                continue

            parts = s.split(",")
            if len(parts) != 5:
                continue

            iters.append(int(parts[0]))
            e_site.append(float(parts[3]))
            e_err.append(float(parts[4]))

    if not iters:
        raise RuntimeError(f"No numeric rows found in {csv_path}")

    return np.array(iters, dtype=int), np.array(e_site, dtype=float), np.array(e_err, dtype=float), meta


def subtitle_from_meta(meta: dict) -> str:
    bits = []
    if "L" in meta and "N_sites" in meta:
        bits.append(f"L={meta['L']} (N={meta['N_sites']})")
    if "J1" in meta:
        bits.append(f"J1={meta['J1']:g}")
    if "J2" in meta:
        bits.append(f"J2={meta['J2']:.2f}")
    if "n_samples" in meta:
        bits.append(f"samples={meta['n_samples']}")
    if "diag_shift" in meta:
        bits.append(f"diag_shift={meta['diag_shift']:g}")
    return " | ".join(bits)


def plot_real_vs_complex(
    outdir: Path,
    J2: float,
    it_r: np.ndarray, e_r: np.ndarray, err_r: np.ndarray,
    it_c: np.ndarray, e_c: np.ndarray, err_c: np.ndarray,
    meta_any: dict,
) -> None:
    """
    Make two sizes (small/big) for a single J2.
    Uses common min length if arrays differ.
    """
    ensure_dir(outdir)
    set_paper_style()

    # Ensure equal length if something differs
    Lmin = min(len(it_r), len(it_c))
    it_r, e_r, err_r = it_r[:Lmin], e_r[:Lmin], err_r[:Lmin]
    it_c, e_c, err_c = it_c[:Lmin], e_c[:Lmin], err_c[:Lmin]

    lit = LITERATURE_E_SITE.get(J2, None)
    subtitle = subtitle_from_meta(meta_any)

    for tag, figsize in [("small", FIG_SMALL), ("big", FIG_BIG)]:
        fig, ax = plt.subplots(figsize=figsize)

        # keep title/subtitle separated (no overlap)
        fig.subplots_adjust(top=0.76)

        # REAL
        (line_r,) = ax.plot(
            it_r, e_r,
            marker="o",
            markersize=2.2 if tag == "small" else 2.6,
            label="RBM real: $E/N \\pm \\sigma/N$",
            zorder=3,
        )
        cr = line_r.get_color()
        ax.errorbar(it_r, e_r, yerr=err_r, fmt="none",
                    ecolor=cr,
                    elinewidth=1.0 if tag == "small" else 1.2,
                    capsize=2.2 if tag == "small" else 2.6,
                    alpha=0.95, zorder=2)
        ax.fill_between(it_r, e_r - err_r, e_r + err_r, color=cr, alpha=0.10, zorder=1)

        # COMPLEX
        (line_c,) = ax.plot(
            it_c, e_c,
            marker="o",
            markersize=2.2 if tag == "small" else 2.6,
            label="RBM complex: $E/N \\pm \\sigma/N$",
            zorder=3,
        )
        cc = line_c.get_color()
        ax.errorbar(it_c, e_c, yerr=err_c, fmt="none",
                    ecolor=cc,
                    elinewidth=1.0 if tag == "small" else 1.2,
                    capsize=2.2 if tag == "small" else 2.6,
                    alpha=0.95, zorder=2)
        ax.fill_between(it_c, e_c - err_c, e_c + err_c, color=cc, alpha=0.10, zorder=1)

        # Literature value line (black)
        if lit is not None:
            ax.axhline(
                lit,
                color="black",
                linestyle="--",
                linewidth=2.2 if tag == "small" else 2.4,
                label="Literature value",
                zorder=0,
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"Energy per site $E/N$")

        fig.suptitle(f"RBM convergence: real vs complex (J2={J2:.2f})", y=0.96,
                     fontsize=15 if tag == "small" else 16)
        fig.text(0.5, 0.88, subtitle, ha="center", va="center",
                 fontsize=11 if tag == "small" else 12)

        ax.minorticks_on()
        ax.tick_params(which="both", direction="out")
        ax.legend(loc="best", handlelength=2.8, borderpad=0.6, labelspacing=0.4)

        fig.savefig(outdir / f"compare_real_vs_complex_J2_{J2:.2f}_{tag}.png")
        fig.savefig(outdir / f"compare_real_vs_complex_J2_{J2:.2f}_{tag}.pdf")
        plt.close(fig)


def main() -> None:
    root = Path(".").resolve()

    real_root = root / "RBM_REAL"
    cplx_root = root / "RBM_COMPLEX"

    if not real_root.exists() or not cplx_root.exists():
        raise SystemExit(
            "Run this script from results_rbm_real_vs_complex/ where RBM_REAL/ and RBM_COMPLEX/ exist."
        )

    out_root = root / "compare_per_J2_clean"
    ensure_dir(out_root)

    for J2 in J2_LIST:
        real_csv = real_root / f"J2_{J2:.2f}" / f"rbm_real_J2_{J2:.2f}_history.csv"
        cplx_csv = cplx_root / f"J2_{J2:.2f}" / f"rbm_complex_J2_{J2:.2f}_history.csv"

        if not real_csv.exists():
            raise FileNotFoundError(f"Missing: {real_csv}")
        if not cplx_csv.exists():
            raise FileNotFoundError(f"Missing: {cplx_csv}")

        it_r, e_r, err_r, meta_r = load_history_csv(real_csv)
        it_c, e_c, err_c, meta_c = load_history_csv(cplx_csv)

        # Use meta from real if present, else complex
        meta_any = meta_r if meta_r else meta_c

        j2_out = out_root / f"J2_{J2:.2f}"
        plot_real_vs_complex(
            outdir=j2_out,
            J2=J2,
            it_r=it_r, e_r=e_r, err_r=err_r,
            it_c=it_c, e_c=e_c, err_c=err_c,
            meta_any=meta_any,
        )

    print("✓ Done. Created per-J2 REAL vs COMPLEX plots (two sizes) in:")
    print(f"  {out_root}")


if __name__ == "__main__":
    main()
