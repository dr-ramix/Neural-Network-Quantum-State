#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Depth-sweep plotting ONLY (no training). Restricted to depth = 1,2,3.

Creates:
1) Real-only:   depth 1 vs 2 vs 3 (real)
2) Complex-only:depth 1 vs 2 vs 3 (complex)
3) All together: (real+complex) for depth 1,2,3

Features:
- error bars for EVERY point (from CSV energy_per_site_sigma)
- error color matches curve color
- subtle error band
- literature value (black dashed) chosen from hard-coded table using J2 from CSV header
- clean title + subtitle (no overlap)
- two sizes: small + big
- saves PNG + PDF
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List, Optional
import re

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Literature values (energy per site)
# -----------------------------
LITERATURE_E_SITE = {
    0.4: -0.52975,
    0.5: -0.50381,
    0.6: -0.49518,
    1.0: -0.71436,
}

DEPTHS = [1, 2, 3]
DTYPES = ["real", "complex"]

FIG_SMALL = (7.2, 4.2)
FIG_BIG   = (9.6, 5.6)


def set_style() -> None:
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
    Reads:
      iter,energy_mean,energy_sigma,energy_per_site_mean,energy_per_site_sigma
    Parses header comments for meta (L, N_sites, J1, J2, n_samples, etc.) when present.
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

            if s.startswith("#"):
                # Example: "# L=6 (N_sites=36) | J1=1.0 | J2=0.5"
                if "L=" in s and "J2=" in s:
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
                # Example: "# n_samples=8000 | n_iter=600 | diag_shift=0.01 | seed=1234"
                if "n_samples=" in s:
                    try:
                        meta["n_samples"] = int(s.split("n_samples=")[1].split()[0])
                    except Exception:
                        pass
                if "diag_shift=" in s:
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
        raise RuntimeError(f"No data rows in {csv_path}")

    return np.array(iters), np.array(e_site), np.array(e_err), meta


def common_length(curves: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> int:
    return min(len(it) for (it, _, _) in curves)


def make_subtitle(meta: dict, extra: str = "") -> str:
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
    if extra:
        bits.append(extra)
    return " | ".join(bits)


def plot(
    outbase: Path,
    title: str,
    subtitle: str,
    curves: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    meta_for_lit: dict,
    figsize: Tuple[float, float],
) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(top=0.76)

    # Plot lines + errors
    for label, it, y, yerr in curves:
        (line,) = ax.plot(it, y, marker="o", markersize=2.2 if figsize[0] < 9 else 2.6,
                          label=label + r": $E/N \pm \sigma/N$", zorder=3)
        c = line.get_color()

        ax.errorbar(it, y, yerr=yerr, fmt="none",
                    ecolor=c,
                    elinewidth=1.0 if figsize[0] < 9 else 1.2,
                    capsize=2.2 if figsize[0] < 9 else 2.6,
                    alpha=0.95, zorder=2)

        ax.fill_between(it, y - yerr, y + yerr, color=c, alpha=0.10, zorder=1)

    # Literature line (black dashed) if J2 matches table
    J2 = meta_for_lit.get("J2", None)
    lit = LITERATURE_E_SITE.get(J2, None) if J2 is not None else None
    if lit is not None:
        ax.axhline(lit, color="black", linestyle="--",
                   linewidth=2.2 if figsize[0] < 9 else 2.4,
                   label="Literature value", zorder=0)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Energy per site $E/N$")

    fig.suptitle(title, y=0.96, fontsize=15 if figsize[0] < 9 else 16)
    fig.text(0.5, 0.88, subtitle, ha="center", va="center", fontsize=11 if figsize[0] < 9 else 12)

    ax.minorticks_on()
    ax.tick_params(which="both", direction="out")
    ax.legend(loc="best", handlelength=2.8, borderpad=0.6, labelspacing=0.4)

    ensure_dir(outbase.parent)
    fig.savefig(outbase.with_suffix(".png"))
    fig.savefig(outbase.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    root = Path(".").resolve()

    # Input folders (fixed by your structure)
    real_root = root / "MLP_depth_real"
    cplx_root = root / "MLP_depth_complex"

    if not real_root.exists() or not cplx_root.exists():
        raise SystemExit(
            "Run this script from inside depth_sweep_gpu_L6_test2_8k/ "
            "where MLP_depth_real/ and MLP_depth_complex/ exist."
        )

    # Output folders (match your project layout)
    out_real = root / "compare_depth_per_dtype" / "real"
    out_cplx = root / "compare_depth_per_dtype" / "complex"
    out_all  = root / "compare_all_depths_and_dtypes"
    ensure_dir(out_real); ensure_dir(out_cplx); ensure_dir(out_all)

    # Load all required runs
    data: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray, np.ndarray, dict]] = {}

    for dtype in DTYPES:
        base = real_root if dtype == "real" else cplx_root
        for depth in DEPTHS:
            csv = base / f"depth_{depth}" / f"MLP_{dtype}_depth_{depth}_history.csv"
            if not csv.exists():
                raise FileNotFoundError(f"Missing CSV: {csv}")
            data[(dtype, depth)] = load_history_csv(csv)

    # Use meta from any run (assumed consistent J2/L/etc.)
    meta_any = data[("real", 1)][3] if ("real", 1) in data else data[("complex", 1)][3]

    # ---- REAL-only plot (depth 1/2/3) ----
    real_curves_raw = [(data[("real", d)][0], data[("real", d)][1], data[("real", d)][2]) for d in DEPTHS]
    Lmin = common_length(real_curves_raw)

    curves_real = []
    for d in DEPTHS:
        it, y, yerr, _ = data[("real", d)]
        curves_real.append((f"depth={d}", it[:Lmin], y[:Lmin], yerr[:Lmin]))

    subtitle_real = make_subtitle(meta_any, extra="dtype=real | depths=1,2,3")
    for name, size in [("small", FIG_SMALL), ("big", FIG_BIG)]:
        plot(
            outbase=out_real / f"compare_depth_1_2_3_real_{name}",
            title="VMC Energy per Site Convergence (depth comparison)",
            subtitle=subtitle_real,
            curves=curves_real,
            meta_for_lit=meta_any,
            figsize=size,
        )

    # ---- COMPLEX-only plot (depth 1/2/3) ----
    cplx_curves_raw = [(data[("complex", d)][0], data[("complex", d)][1], data[("complex", d)][2]) for d in DEPTHS]
    Lmin = common_length(cplx_curves_raw)

    curves_cplx = []
    for d in DEPTHS:
        it, y, yerr, _ = data[("complex", d)]
        curves_cplx.append((f"depth={d}", it[:Lmin], y[:Lmin], yerr[:Lmin]))

    subtitle_cplx = make_subtitle(meta_any, extra="dtype=complex | depths=1,2,3")
    for name, size in [("small", FIG_SMALL), ("big", FIG_BIG)]:
        plot(
            outbase=out_cplx / f"compare_depth_1_2_3_complex_{name}",
            title="VMC Energy per Site Convergence (depth comparison)",
            subtitle=subtitle_cplx,
            curves=curves_cplx,
            meta_for_lit=meta_any,
            figsize=size,
        )

    # ---- ALL together plot (real+complex, depth 1/2/3) ----
    all_curves_raw = []
    for dtype in DTYPES:
        for d in DEPTHS:
            it, y, yerr, _ = data[(dtype, d)]
            all_curves_raw.append((it, y, yerr))
    Lmin = common_length(all_curves_raw)

    curves_all = []
    for dtype in ("real", "complex"):
        for d in DEPTHS:
            it, y, yerr, _ = data[(dtype, d)]
            curves_all.append((f"{dtype}, depth={d}", it[:Lmin], y[:Lmin], yerr[:Lmin]))

    subtitle_all = make_subtitle(meta_any, extra="all dtypes | depths=1,2,3")
    for name, size in [("small", FIG_SMALL), ("big", FIG_BIG)]:
        plot(
            outbase=out_all / f"compare_all_depths_and_dtypes_1_2_3_{name}",
            title="VMC Energy per Site Convergence (all depths and dtypes)",
            subtitle=subtitle_all,
            curves=curves_all,
            meta_for_lit=meta_any,
            figsize=size,
        )

    print("✓ Done. Created plots (depth 1,2,3 only):")
    print("  - compare_depth_per_dtype/real/compare_depth_1_2_3_real_{small,big}.(png|pdf)")
    print("  - compare_depth_per_dtype/complex/compare_depth_1_2_3_complex_{small,big}.(png|pdf)")
    print("  - compare_all_depths_and_dtypes/compare_all_depths_and_dtypes_1_2_3_{small,big}.(png|pdf)")


if __name__ == "__main__":
    main()
