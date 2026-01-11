#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare RBM with REAL vs COMPLEX parameters for the J1-J2 Heisenberg model on a square lattice (PBC).

Experiment:
  - J1 fixed = 1.0
  - J2 in [0.4, 0.5, 0.6, 1.0]
  - LxL square lattice, total_sz = 0
  - RBM + VMC(SR) with MetropolisExchange sampler

Requested outputs:
  - High-resolution plots (professional style)
  - Informative titles
  - CSV files with header "title" lines (metadata at top)
  - Summary TXT file more readable
  - For *every plot type* produce 3 variants:
      (a) no true value line
      (b) with paper true value line (provided below)
      (c) with NetKet ED true value line if feasible (small systems only)

Notes:
  - For L=6 => N=36, ED is typically NOT feasible. We still try it only if N_sites <= --ed_max_sites.
  - Learning rate is fixed to 1e-3
  - Iterations fixed to 800

Usage examples:
  python compare_rbm_real_vs_complex.py --platform gpu
  python compare_rbm_real_vs_complex.py --platform cpu --L 4 --ed_max_sites 16
"""

import os
import json
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Platform selection MUST happen before importing jax/netket
# -----------------------------
def preparse_platform(argv: List[str]) -> str:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--platform", type=str, default="auto",
                   choices=["auto", "cpu", "gpu", "tpu"])
    args, _ = p.parse_known_args(argv)
    return args.platform


def set_platform(platform: str) -> None:
    if platform.lower() != "auto":
        os.environ["JAX_PLATFORM_NAME"] = platform.lower()


_platform = preparse_platform(os.sys.argv[1:])
set_platform(_platform)

# Now safe to import jax/netket
import jax
import jax.numpy as jnp
import netket as nk


# -----------------------------
# Paper "true" values provided by you (energy per site)
# -----------------------------
PAPER_TRUE_E_SITE: Dict[float, float] = {
    0.4: -0.52975,
    0.5: -0.50381,
    0.6: -0.49518,
    1.0: -0.71436,
}


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def fmt_pm(val: float, err: float, prec: int = 6) -> str:
    return f"{val:.{prec}f} ± {err:.{prec}f}"


def style_matplotlib():
    # High-resolution + clean style
    plt.rcParams.update({
        "figure.dpi": 220,
        "savefig.dpi": 450,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 2.0,
    })


def make_lattice_and_hamiltonian(L: int, J1: float, J2: float):
    lattice = nk.graph.Square(length=L, max_neighbor_order=2, pbc=True)
    hilbert = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=lattice.n_nodes)

    # For Square(max_neighbor_order=2): J=[J1, J2] is NN and NNN couplings.
    ham = nk.operator.Heisenberg(
        hilbert=hilbert,
        graph=lattice,
        J=[J1, J2],
        sign_rule=[False, False],
    )
    return lattice, hilbert, ham


def build_sampler(hilbert, lattice, d_max: int = 2):
    return nk.sampler.MetropolisExchange(hilbert=hilbert, graph=lattice, d_max=d_max)


def build_rbm(alpha: int, param_dtype: Any):
    return nk.models.RBM(
        alpha=alpha,
        use_hidden_bias=True,
        use_visible_bias=True,
        param_dtype=param_dtype,
    )


def build_vmc_sr_driver(hamiltonian, vstate, optimizer, diag_shift: float):
    """
    NetKet API compatibility:
      - If nk.driver.VMC_SR exists, use it.
      - Otherwise use nk.driver.VMC with SR preconditioner (newer API).
    """
    if hasattr(nk.driver, "VMC_SR"):
        return nk.driver.VMC_SR(
            hamiltonian=hamiltonian,
            optimizer=optimizer,
            diag_shift=diag_shift,
            variational_state=vstate,
        )

    if hasattr(nk.optimizer, "SR"):
        sr = nk.optimizer.SR(diag_shift=diag_shift)
        return nk.driver.VMC(
            hamiltonian=hamiltonian,
            optimizer=optimizer,
            variational_state=vstate,
            preconditioner=sr,
        )

    raise RuntimeError(
        "Could not find SR driver/preconditioner. "
        "Your NetKet version may be too old/new for this script."
    )


def maybe_exact_ground_state_energy_per_site(
    L: int,
    J1: float,
    J2: float,
    ed_max_sites: int,
) -> Optional[float]:
    """
    Try to compute an ED/Lanczos ground-state energy per site with NetKet,
    but ONLY if N_sites <= ed_max_sites.
    """
    lattice, hilbert, ham = make_lattice_and_hamiltonian(L, J1, J2)
    n_sites = lattice.n_nodes
    if n_sites > ed_max_sites:
        return None

    try:
        # Prefer Lanczos if available
        if hasattr(nk.exact, "lanczos_ed"):
            res = nk.exact.lanczos_ed(ham, k=1, compute_eigenvectors=False)
            # Handle possible return shapes across versions
            if hasattr(res, "eigenvalues"):
                e0 = float(np.asarray(res.eigenvalues)[0])
            elif isinstance(res, dict) and "eigenvalues" in res:
                e0 = float(np.asarray(res["eigenvalues"])[0])
            else:
                arr = np.asarray(res[0] if isinstance(res, (tuple, list)) else res)
                e0 = float(arr[0])
            return e0 / n_sites

        # Fallback to full diagonalization if present (small only)
        if hasattr(nk.exact, "diag"):
            evals = nk.exact.diag(ham)
            e0 = float(np.min(np.asarray(evals)))
            return e0 / n_sites

    except Exception:
        return None

    return None


def _plot_energy_curve(
    outpath: Path,
    title: str,
    iters: np.ndarray,
    curves: List[Tuple[str, np.ndarray, np.ndarray]],
    true_line: Optional[Tuple[str, float]] = None,
):
    style_matplotlib()
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    for label, y, yerr in curves:
        ax.plot(iters, y, label=label)
        ax.fill_between(iters, y - yerr, y + yerr, alpha=0.20)

    if true_line is not None:
        tlab, tval = true_line
        ax.axhline(tval, linestyle="--", linewidth=2.0, label=tlab)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy per site")
    ax.set_title(title)
    ax.legend(loc="best")

    fig.tight_layout()
    ensure_dir(outpath.parent)
    fig.savefig(outpath)
    plt.close(fig)


def plot_three_variants(
    outdir: Path,
    base_name: str,
    title_base: str,
    iters: np.ndarray,
    curves: List[Tuple[str, np.ndarray, np.ndarray]],
    paper_true: Optional[float],
    ed_true: Optional[float],
):
    """
    Writes:
      - base_name.png
      - base_name__paper_true.png (if paper_true available)
      - base_name__ed_true.png (if ed_true available)
    """
    _plot_energy_curve(
        outpath=outdir / f"{base_name}.png",
        title=title_base,
        iters=iters,
        curves=curves,
        true_line=None,
    )

    if paper_true is not None:
        _plot_energy_curve(
            outpath=outdir / f"{base_name}__paper_true.png",
            title=f"{title_base} | Paper reference",
            iters=iters,
            curves=curves,
            true_line=("Paper E/site", float(paper_true)),
        )

    if ed_true is not None:
        _plot_energy_curve(
            outpath=outdir / f"{base_name}__ed_true.png",
            title=f"{title_base} | NetKet ED reference",
            iters=iters,
            curves=curves,
            true_line=("NetKet ED E/site", float(ed_true)),
        )


# -----------------------------
# Config
# -----------------------------
@dataclass
class RunConfig:
    L: int = 6
    J1: float = 1.0
    J2: float = 0.5

    n_samples: int = 10000
    n_discard_per_chain: int = 50
    n_iter: int = 800          # requested
    diag_shift: float = 0.01
    seed: int = 1234

    # Sampler
    d_max: int = 2

    # RBM
    rbm_alpha: int = 4
    rbm_lr: float = 1e-3       # requested


def run_rbm_vmc_sr(
    cfg: RunConfig,
    outdir: Path,
    *,
    param_dtype: Any,
    dtype_label: str,
    paper_true: Optional[float],
    ed_true: Optional[float],
) -> Dict[str, np.ndarray]:
    """
    One RBM optimization run, with:
      - history CSV (with metadata lines at top)
      - run_meta.json
      - training curve PNG (3 variants)
    """
    ensure_dir(outdir)

    lattice, hilbert, ham = make_lattice_and_hamiltonian(cfg.L, cfg.J1, cfg.J2)
    n_sites = lattice.n_nodes

    sampler = build_sampler(hilbert, lattice, cfg.d_max)
    model = build_rbm(cfg.rbm_alpha, param_dtype=param_dtype)
    opt = nk.optimizer.Adam(learning_rate=cfg.rbm_lr)

    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=cfg.n_samples,
        n_discard_per_chain=cfg.n_discard_per_chain,
        seed=cfg.seed,
    )

    driver = build_vmc_sr_driver(hamiltonian=ham, vstate=vstate, optimizer=opt, diag_shift=cfg.diag_shift)
    log = nk.logging.RuntimeLog()

    t0 = time.time()
    driver.run(n_iter=cfg.n_iter, out=log)
    t1 = time.time()

    E_hist = log.data["Energy"]
    iters = np.asarray(E_hist.iters, dtype=int)
    E_mean = np.asarray(E_hist.Mean.real, dtype=float)
    E_sigma = np.asarray(E_hist.Sigma.real, dtype=float)

    e_site = E_mean / n_sites
    e_site_err = E_sigma / n_sites

    # ---- History CSV (with metadata "title" lines) ----
    csv_path = outdir / f"rbm_{dtype_label.lower()}_J2_{cfg.J2:.2f}_history.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(f"# RBM history | dtype={dtype_label} | L={cfg.L} (N_sites={n_sites}) | J1={cfg.J1} | J2={cfg.J2}\n")
        f.write(f"# Optimizer=Adam | lr={cfg.rbm_lr} | alpha={cfg.rbm_alpha} | SR diag_shift={cfg.diag_shift}\n")
        f.write(f"# Sampler=MetropolisExchange(d_max={cfg.d_max}) | n_samples={cfg.n_samples} | discard={cfg.n_discard_per_chain} | seed={cfg.seed}\n")
        if paper_true is not None:
            f.write(f"# Paper reference E/site = {paper_true:.5f}\n")
        if ed_true is not None:
            f.write(f"# NetKet ED reference E/site = {ed_true:.12f}\n")
        f.write("iter,energy_mean,energy_sigma,energy_per_site_mean,energy_per_site_sigma\n")
        for i, em, es, eps, epe in zip(iters, E_mean, E_sigma, e_site, e_site_err):
            f.write(f"{int(i)},{em:.12f},{es:.12f},{eps:.12f},{epe:.12f}\n")

    # ---- Meta JSON ----
    meta = {
        "ansatz": "RBM",
        "dtype_label": dtype_label,
        "L": cfg.L,
        "n_sites": int(n_sites),
        "J1": cfg.J1,
        "J2": cfg.J2,
        "n_samples": cfg.n_samples,
        "n_discard_per_chain": cfg.n_discard_per_chain,
        "n_iter": cfg.n_iter,
        "diag_shift": cfg.diag_shift,
        "seed": cfg.seed,
        "rbm_alpha": cfg.rbm_alpha,
        "optimizer": {"type": "Adam", "learning_rate": cfg.rbm_lr},
        "param_dtype": str(param_dtype),
        "paper_true_e_site": paper_true,
        "ed_true_e_site": ed_true,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "n_parameters": int(vstate.n_parameters),
        "runtime_seconds": float(t1 - t0),
        "final_energy_per_site": float(e_site[-1]),
        "final_energy_per_site_err": float(e_site_err[-1]),
    }
    save_json(outdir / "run_meta.json", meta)

    # ---- Training curve plot (3 variants) ----
    title_base = (
        f"RBM ({dtype_label} params) | L={cfg.L} (N_sites={n_sites}) | "
        f"J1={cfg.J1} | J2={cfg.J2} | samples={cfg.n_samples} | iters={cfg.n_iter} | lr={cfg.rbm_lr}"
    )
    plot_three_variants(
        outdir=outdir,
        base_name=f"rbm_{dtype_label.lower()}_J2_{cfg.J2:.2f}_energy_per_site",
        title_base=title_base,
        iters=iters,
        curves=[(f"RBM {dtype_label} (E/site)", e_site, e_site_err)],
        paper_true=paper_true,
        ed_true=ed_true,
    )

    return {
        "iters": iters,
        "e_site": e_site,
        "e_site_err": e_site_err,
        "n_sites": np.array([n_sites], dtype=np.int64),
        "runtime_s": np.array([t1 - t0], dtype=np.float64),
        "n_params": np.array([int(vstate.n_parameters)], dtype=np.int64),
    }


def plot_real_vs_complex_for_j2_three_variants(
    outdir: Path,
    *,
    L: int,
    n_sites: int,
    J1: float,
    J2: float,
    n_samples: int,
    n_iter: int,
    lr: float,
    hist_real: Dict[str, np.ndarray],
    hist_cplx: Dict[str, np.ndarray],
    paper_true: Optional[float],
    ed_true: Optional[float],
):
    ensure_dir(outdir)

    title_base = (
        f"RBM Real vs Complex | L={L} (N_sites={n_sites}) | J1={J1} | J2={J2} | "
        f"samples={n_samples} | iters={n_iter} | lr={lr}"
    )

    # Use iteration axis from real (both have same n_iter)
    iters = hist_real["iters"]
    curves = [
        ("RBM Real (E/site)",    hist_real["e_site"], hist_real["e_site_err"]),
        ("RBM Complex (E/site)", hist_cplx["e_site"], hist_cplx["e_site_err"]),
    ]

    base_name = f"compare_real_vs_complex_J2_{J2:.2f}"
    plot_three_variants(
        outdir=outdir,
        base_name=base_name,
        title_base=title_base,
        iters=iters,
        curves=curves,
        paper_true=paper_true,
        ed_true=ed_true,
    )


def plot_final_summary_vs_j2_three_variants(
    outdir: Path,
    *,
    J2_list: List[float],
    real_final: np.ndarray,
    real_final_err: np.ndarray,
    cplx_final: np.ndarray,
    cplx_final_err: np.ndarray,
    paper_true_map: Dict[float, float],
    ed_true_map: Dict[float, Optional[float]],
    L: int,
    n_sites: int,
    J1: float,
    n_samples: int,
    n_iter: int,
    lr: float,
):
    ensure_dir(outdir)
    style_matplotlib()

    # Prepare paper/ed lines: only plot if all values exist for those J2s (paper does here)
    paper_vals = np.array([paper_true_map.get(j2, np.nan) for j2 in J2_list], dtype=float)
    have_paper = np.all(np.isfinite(paper_vals))

    ed_vals = np.array([np.nan if ed_true_map.get(j2, None) is None else float(ed_true_map[j2]) for j2 in J2_list], dtype=float)
    have_ed = np.all(np.isfinite(ed_vals))

    def _make(outpath: Path, title: str, draw_paper: bool, draw_ed: bool):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)

        ax.errorbar(J2_list, real_final, yerr=real_final_err, fmt="o-", label="RBM Real final E/site")
        ax.errorbar(J2_list, cplx_final, yerr=cplx_final_err, fmt="o-", label="RBM Complex final E/site")

        if draw_paper:
            ax.plot(J2_list, paper_vals, linestyle="--", linewidth=2.0, label="Paper E/site")

        if draw_ed:
            ax.plot(J2_list, ed_vals, linestyle="--", linewidth=2.0, label="NetKet ED E/site")

        ax.set_xlabel("J2")
        ax.set_ylabel("Final energy per site")
        ax.set_title(title)
        ax.legend(loc="best")

        fig.tight_layout()
        fig.savefig(outpath)
        plt.close(fig)

    title_base = (
        f"Final Energy per Site vs J2 | RBM Real vs Complex | "
        f"L={L} (N_sites={n_sites}) | J1={J1} | samples={n_samples} | iters={n_iter} | lr={lr}"
    )

    _make(outdir / "final_E_site_vs_J2_real_vs_complex.png", title_base, draw_paper=False, draw_ed=False)

    if have_paper:
        _make(outdir / "final_E_site_vs_J2_real_vs_complex__paper_true.png",
              f"{title_base} | Paper reference", draw_paper=True, draw_ed=False)

    if have_ed:
        _make(outdir / "final_E_site_vs_J2_real_vs_complex__ed_true.png",
              f"{title_base} | NetKet ED reference", draw_paper=False, draw_ed=True)


def write_summary_files(outdir: Path, rows: List[Dict[str, Any]]):
    """
    summary.csv: one row per run (dtype per J2)
    summary.txt: readable fixed-width table
    """
    ensure_dir(outdir)

    csv_path = outdir / "summary.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("# Summary of RBM Real vs Complex runs (final energies per site)\n")
        f.write("# One row per run: (dtype, J2)\n")
        f.write("dtype,J2,final_energy_per_site,final_energy_per_site_err,n_parameters,runtime_seconds,paper_true_e_site,ed_true_e_site\n")
        for r in rows:
            f.write(
                f"{r['dtype']},{r['J2']:.2f},"
                f"{r['final_e_site']:.12f},{r['final_e_site_err']:.12f},"
                f"{r['n_params']},{r['runtime_s']:.6f},"
                f"{'' if r['paper_true'] is None else f'{r['paper_true']:.6f}'},"
                f"{'' if r['ed_true'] is None else f'{r['ed_true']:.12f}'}\n"
            )

    headers = ["DTYPE", "J2", "FINAL E/SITE", "ERR", "N_PARAMS", "RUNTIME(s)", "PAPER", "ED"]
    table = []
    for r in rows:
        table.append([
            r["dtype"],
            f"{r['J2']:.2f}",
            f"{r['final_e_site']:.8f}",
            f"{r['final_e_site_err']:.8f}",
            str(r["n_params"]),
            f"{r['runtime_s']:.2f}",
            ("" if r["paper_true"] is None else f"{r['paper_true']:.5f}"),
            ("" if r["ed_true"] is None else f"{r['ed_true']:.8f}"),
        ])

    colw = [len(h) for h in headers]
    for row in table:
        for i, cell in enumerate(row):
            colw[i] = max(colw[i], len(cell))

    def line(ch="-"):
        return "+" + "+".join([ch * (w + 2) for w in colw]) + "+\n"

    txt_path = outdir / "summary.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("RBM Real vs Complex — Final Energies per Site\n")
        f.write(line("-"))
        f.write("| " + " | ".join([headers[i].ljust(colw[i]) for i in range(len(headers))]) + " |\n")
        f.write(line("="))
        for row in table:
            f.write("| " + " | ".join([row[i].ljust(colw[i]) for i in range(len(headers))]) + " |\n")
        f.write(line("-"))


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="RBM Real vs Complex comparison for J1-J2 Heisenberg on square lattice (NetKet)."
    )
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--J1", type=float, default=1.0)
    parser.add_argument("--J2_list", type=str, default="0.4,0.5,0.6,1.0")

    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_iter", type=int, default=800)           # requested
    parser.add_argument("--discard", type=int, default=50)
    parser.add_argument("--diag_shift", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--rbm_alpha", type=int, default=4)
    parser.add_argument("--rbm_lr", type=float, default=1e-3)        # requested (fixed default)

    parser.add_argument("--d_max", type=int, default=2)

    parser.add_argument("--platform", type=str, default=_platform,
                        choices=["auto", "cpu", "gpu", "tpu"],
                        help="JAX platform selection. Use 'auto' to let JAX decide.")

    parser.add_argument("--out", type=str, default="results_rbm_real_vs_complex")

    # ED reference control
    parser.add_argument("--ed_max_sites", type=int, default=20,
                        help="Compute NetKet ED reference only if N_sites <= ed_max_sites.")
    args = parser.parse_args()

    # Parse J2 list
    J2_list = [float(x.strip()) for x in args.J2_list.split(",") if x.strip() != ""]
    out_root = Path(args.out)
    ensure_dir(out_root)

    n_sites = args.L * args.L

    print("\n===================================================")
    print("RBM Real vs Complex (VMC_SR)")
    print("===================================================")
    print(f"Requested platform: {args.platform}")
    print(f"JAX backend:        {jax.default_backend()}")
    print("JAX devices:        " + ", ".join([str(d) for d in jax.devices()]))
    print("---------------------------------------------------")
    print(f"L={args.L} -> N_sites={n_sites}")
    print(f"J1={args.J1}")
    print(f"J2_list={J2_list}")
    print("---------------------------------------------------")
    print(f"n_samples={args.n_samples} | n_iter={args.n_iter} | discard={args.discard} | diag_shift={args.diag_shift} | seed={args.seed}")
    print(f"RBM: alpha={args.rbm_alpha} | lr={args.rbm_lr}")
    print(f"ED reference: enabled only if N_sites <= {args.ed_max_sites}")
    print("===================================================\n")

    # Compute ED references per J2 (most will be None for L=6)
    ed_true_map: Dict[float, Optional[float]] = {}
    for J2 in J2_list:
        ed_true_map[J2] = maybe_exact_ground_state_energy_per_site(args.L, args.J1, J2, args.ed_max_sites)

    # Save sweep config
    save_json(out_root / "sweep_config.json", {
        "L": args.L,
        "J1": args.J1,
        "J2_list": J2_list,
        "n_samples": args.n_samples,
        "n_iter": args.n_iter,
        "discard": args.discard,
        "diag_shift": args.diag_shift,
        "seed": args.seed,
        "rbm": {"alpha": args.rbm_alpha, "lr": args.rbm_lr},
        "compare": {"real_param_dtype": "float64", "complex_param_dtype": "complex128"},
        "paper_true_e_site": {str(k): v for k, v in PAPER_TRUE_E_SITE.items()},
        "ed_true_e_site": {str(k): (None if ed_true_map[k] is None else float(ed_true_map[k])) for k in ed_true_map},
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
    })

    # Collect final values
    summary_rows: List[Dict[str, Any]] = []
    real_final, real_final_err = [], []
    cplx_final, cplx_final_err = [], []

    compare_dir = out_root / "compare_per_J2"
    ensure_dir(compare_dir)

    for J2 in J2_list:
        paper_true = PAPER_TRUE_E_SITE.get(J2, None)
        ed_true = ed_true_map.get(J2, None)

        print(f"=== J2 = {J2:.2f} ===")
        if paper_true is not None:
            print(f"Paper reference E/site: {paper_true:.6f}")
        else:
            print("Paper reference E/site: (not provided)")
        if ed_true is None:
            print("NetKet ED reference:    (skipped / not feasible)")
        else:
            print(f"NetKet ED reference:    {ed_true:.12f}")

        cfg = RunConfig(
            L=args.L,
            J1=args.J1,
            J2=J2,
            n_samples=args.n_samples,
            n_discard_per_chain=args.discard,
            n_iter=args.n_iter,
            diag_shift=args.diag_shift,
            seed=args.seed,
            d_max=args.d_max,
            rbm_alpha=args.rbm_alpha,
            rbm_lr=args.rbm_lr,
        )

        # REAL parameters
        real_dir = out_root / "RBM_REAL" / f"J2_{J2:.2f}"
        print(f"[RUN] RBM REAL    -> {real_dir}")
        hist_real = run_rbm_vmc_sr(
            cfg,
            real_dir,
            param_dtype=jnp.float64,
            dtype_label="REAL",
            paper_true=paper_true,
            ed_true=ed_true,
        )

        # COMPLEX parameters
        cplx_dir = out_root / "RBM_COMPLEX" / f"J2_{J2:.2f}"
        print(f"[RUN] RBM COMPLEX -> {cplx_dir}")
        hist_cplx = run_rbm_vmc_sr(
            cfg,
            cplx_dir,
            param_dtype=jnp.complex128,
            dtype_label="COMPLEX",
            paper_true=paper_true,
            ed_true=ed_true,
        )

        # Per-J2 comparison plot (3 variants)
        plot_real_vs_complex_for_j2_three_variants(
            compare_dir,
            L=args.L,
            n_sites=n_sites,
            J1=args.J1,
            J2=J2,
            n_samples=args.n_samples,
            n_iter=args.n_iter,
            lr=args.rbm_lr,
            hist_real=hist_real,
            hist_cplx=hist_cplx,
            paper_true=paper_true,
            ed_true=ed_true,
        )

        # Final numbers (and terminal deltas)
        real_e = float(hist_real["e_site"][-1])
        real_eerr = float(hist_real["e_site_err"][-1])
        cplx_e = float(hist_cplx["e_site"][-1])
        cplx_eerr = float(hist_cplx["e_site_err"][-1])

        real_final.append(real_e)
        real_final_err.append(real_eerr)
        cplx_final.append(cplx_e)
        cplx_final_err.append(cplx_eerr)

        def d(x: float, ref: Optional[float]) -> Optional[float]:
            return None if ref is None else (x - float(ref))

        print("\n[RESULT] Final energy per site")
        print(f"  RBM Real   : {fmt_pm(real_e, real_eerr)}"
              + ("" if paper_true is None else f" | ΔPaper={d(real_e, paper_true):+.6f}")
              + ("" if ed_true is None else f" | ΔED={d(real_e, ed_true):+.6f}"))
        print(f"  RBM Complex: {fmt_pm(cplx_e, cplx_eerr)}"
              + ("" if paper_true is None else f" | ΔPaper={d(cplx_e, paper_true):+.6f}")
              + ("" if ed_true is None else f" | ΔED={d(cplx_e, ed_true):+.6f}"))
        if paper_true is not None:
            print(f"  Paper true : {paper_true:.6f}")
        if ed_true is not None:
            print(f"  NetKet ED  : {ed_true:.12f}")
        print("")

        # Summary rows (two per J2)
        summary_rows.append({
            "dtype": "REAL",
            "J2": J2,
            "final_e_site": real_e,
            "final_e_site_err": real_eerr,
            "n_params": int(hist_real["n_params"][0]),
            "runtime_s": float(hist_real["runtime_s"][0]),
            "paper_true": paper_true,
            "ed_true": ed_true,
        })
        summary_rows.append({
            "dtype": "COMPLEX",
            "J2": J2,
            "final_e_site": cplx_e,
            "final_e_site_err": cplx_eerr,
            "n_params": int(hist_cplx["n_params"][0]),
            "runtime_s": float(hist_cplx["runtime_s"][0]),
            "paper_true": paper_true,
            "ed_true": ed_true,
        })

    # Write summary
    write_summary_files(out_root, summary_rows)

    # Final summary plot vs J2 (3 variants)
    plot_final_summary_vs_j2_three_variants(
        outdir=out_root,
        J2_list=J2_list,
        real_final=np.array(real_final, dtype=float),
        real_final_err=np.array(real_final_err, dtype=float),
        cplx_final=np.array(cplx_final, dtype=float),
        cplx_final_err=np.array(cplx_final_err, dtype=float),
        paper_true_map=PAPER_TRUE_E_SITE,
        ed_true_map=ed_true_map,
        L=args.L,
        n_sites=n_sites,
        J1=args.J1,
        n_samples=args.n_samples,
        n_iter=args.n_iter,
        lr=args.rbm_lr,
    )

    print("\n===================================================")
    print("DONE")
    print("===================================================")
    print(f"All outputs saved to: {out_root.resolve()}")
    print("Key outputs:")
    print("  - RBM_REAL/J2_*/  and RBM_COMPLEX/J2_*/  (per-run CSV + plots + run_meta.json)")
    print("  - compare_per_J2/ (real vs complex plots per J2, 3 variants each)")
    print("  - final_E_site_vs_J2_real_vs_complex*.png (3 variants)")
    print("  - summary.csv / summary.txt / sweep_config.json")
    print("Note: NetKet ED reference is skipped automatically unless N_sites <= ed_max_sites.\n")


if __name__ == "__main__":
    main()
