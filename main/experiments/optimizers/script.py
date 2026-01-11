#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimizer sweep: Adam vs AdamW for J1-J2 Heisenberg on a square lattice (PBC).

Fixed:
  - J1 = 1.0
  - J2 = 0.5
  - LxL square lattice, total_sz = 0 (same as before)

We compare:
  - Ansatz: MLP (two hidden layers: N,N; activation log_cosh) and RBM
  - Parameter dtype: real and complex
  - Optimizer: Adam and AdamW

So total runs = 2 (ansatz) * 2 (dtype) * 2 (optimizer) = 8 runs.

True values:
  - Paper E/site for J2=0.5: -0.50381
  - NetKet ED E/site if feasible (only if N_sites <= --ed_max_sites)

For every plot we generate 3 variants:
  (a) no true value
  (b) with paper true value
  (c) with NetKet ED true value if feasible

Outputs (under --out):
  MLP_real/
    Adam/, AdamW/
  MLP_complex/
    Adam/, AdamW/
  RBM_real/
    Adam/, AdamW/
  RBM_complex/
    Adam/, AdamW/
  compare/
    - compare_MLP_real_Adam_vs_AdamW*.png (+paper/+ed)
    - compare_MLP_complex_Adam_vs_AdamW*.png (+paper/+ed)
    - compare_RBM_real_Adam_vs_AdamW*.png (+paper/+ed)
    - compare_RBM_complex_Adam_vs_AdamW*.png (+paper/+ed)
    - compare_all_8_runs*.png (+paper/+ed)
  summaries:
    - per_run_summary.csv / per_run_summary.txt       (one row per run)
    - overall_results.csv / overall_results.txt       (side-by-side tables per (ansatz,dtype))
    - sweep_config.json

GPU support:
  --platform gpu sets JAX_PLATFORM_NAME=gpu BEFORE importing jax/netket.

Notes on AdamW:
  NetKet versions differ. This script tries:
    - nk.optimizer.AdamW(...)
  If unavailable, it falls back to optax.adamw via nk.optimizer.Optax (if present).
  If neither exists, it raises a clear error.

Usage:
  python optimizer_sweep_adam_vs_adamw.py --platform gpu
  python optimizer_sweep_adam_vs_adamw.py --platform cpu --L 4 --ed_max_sites 16
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

import jax
import jax.numpy as jnp
import netket as nk
from netket.nn.activation import log_cosh


# -----------------------------
# Paper true value provided by you (energy per site)
# -----------------------------
PAPER_TRUE_E_SITE = -0.50381


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


# -----------------------------
# Model / VMC(SR) helpers
# -----------------------------
def make_lattice_and_hamiltonian(L: int, J1: float, J2: float):
    lattice = nk.graph.Square(length=L, max_neighbor_order=2, pbc=True)
    hilbert = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=lattice.n_nodes)
    ham = nk.operator.Heisenberg(
        hilbert=hilbert,
        graph=lattice,
        J=[J1, J2],
        sign_rule=[False, False],
    )
    return lattice, hilbert, ham

def build_sampler(hilbert, lattice, d_max: int = 2):
    return nk.sampler.MetropolisExchange(hilbert=hilbert, graph=lattice, d_max=d_max)

def build_mlp_model_two_layers(n_sites: int, hidden_scale: int, param_dtype: Any):
    """
    Two hidden layers (N,N) with N = n_sites*hidden_scale, activation = log_cosh.
    """
    h = int(n_sites * hidden_scale)
    return nk.models.MLP(
        hidden_dims=(h, h),
        param_dtype=param_dtype,
        hidden_activations=log_cosh,
        output_activation=None,
        use_output_bias=True,
    )

def build_rbm_model(alpha: int, param_dtype: Any):
    return nk.models.RBM(
        alpha=alpha,
        use_hidden_bias=True,
        use_visible_bias=True,
        param_dtype=param_dtype,
    )

def build_vmc_sr_driver(hamiltonian, vstate, optimizer, diag_shift: float):
    """
    Compatibility across NetKet versions:
      - If nk.driver.VMC_SR exists, use it.
      - Else use nk.driver.VMC with SR preconditioner if available.
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
        "Your NetKet version may be incompatible with this script."
    )


# -----------------------------
# ED reference if feasible
# -----------------------------
def maybe_exact_ground_state_energy_per_site(
    L: int,
    J1: float,
    J2: float,
    ed_max_sites: int,
) -> Optional[float]:
    lattice, hilbert, ham = make_lattice_and_hamiltonian(L, J1, J2)
    n_sites = lattice.n_nodes
    if n_sites > ed_max_sites:
        return None

    try:
        if hasattr(nk.exact, "lanczos_ed"):
            res = nk.exact.lanczos_ed(ham, k=1, compute_eigenvectors=False)
            if hasattr(res, "eigenvalues"):
                e0 = float(np.asarray(res.eigenvalues)[0])
            elif isinstance(res, dict) and "eigenvalues" in res:
                e0 = float(np.asarray(res["eigenvalues"])[0])
            else:
                arr = np.asarray(res[0] if isinstance(res, (tuple, list)) else res)
                e0 = float(arr[0])
            return e0 / n_sites

        if hasattr(nk.exact, "diag"):
            evals = nk.exact.diag(ham)
            e0 = float(np.min(np.asarray(evals)))
            return e0 / n_sites

    except Exception:
        return None

    return None


# -----------------------------
# Optimizers (Adam / AdamW) with version-safe construction
# -----------------------------
def build_optimizer(name: str, lr: float, weight_decay: float):
    """
    Return a NetKet optimizer instance.

    - Adam:
        nk.optimizer.Adam(learning_rate=lr)
    - AdamW:
        try nk.optimizer.AdamW(learning_rate=lr, weight_decay=weight_decay)
        else try optax.adamw + nk.optimizer.Optax
    """
    n = name.lower()
    if n == "adam":
        return nk.optimizer.Adam(learning_rate=lr)

    if n == "adamw":
        # 1) Native NetKet AdamW if present
        if hasattr(nk.optimizer, "AdamW"):
            # Some versions may use argument name weight_decay or b1/b2/eps; we only set common ones.
            try:
                return nk.optimizer.AdamW(learning_rate=lr, weight_decay=weight_decay)
            except TypeError:
                # If signature differs, try without weight_decay
                return nk.optimizer.AdamW(learning_rate=lr)

        # 2) Optax fallback if available
        try:
            import optax  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "AdamW requested, but neither nk.optimizer.AdamW nor optax is available. "
                "Install optax or upgrade NetKet."
            ) from e

        if hasattr(nk.optimizer, "Optax"):
            tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
            return nk.optimizer.Optax(tx)

        raise RuntimeError(
            "AdamW requested, but nk.optimizer.AdamW not found and nk.optimizer.Optax not available. "
            "Upgrade NetKet or use a version that provides Optax wrapper."
        )

    raise ValueError(f"Unknown optimizer '{name}'. Use 'Adam' or 'AdamW'.")


# -----------------------------
# Plotting helpers (3 variants)
# -----------------------------
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

def plot_three_variants_energy_curve(
    outdir: Path,
    base_name: str,
    title_base: str,
    iters: np.ndarray,
    curves: List[Tuple[str, np.ndarray, np.ndarray]],
    paper_true: Optional[float],
    ed_true: Optional[float],
):
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
# Run a single VMC(SR) experiment
# -----------------------------
@dataclass
class RunConfig:
    L: int = 6
    J1: float = 1.0
    J2: float = 0.5

    n_samples: int = 10000
    n_discard_per_chain: int = 50
    n_iter: int = 600
    diag_shift: float = 0.01
    seed: int = 1234

    d_max: int = 2

    # MLP
    mlp_lr: float = 1e-3
    mlp_hidden_scale: int = 1

    # RBM
    rbm_lr: float = 1e-2
    rbm_alpha: int = 4

    # AdamW weight decay
    weight_decay: float = 1e-4

    param_dtype: Any = jnp.float64


def run_single(
    ansatz: str,
    dtype_label: str,
    opt_name: str,
    cfg: RunConfig,
    outdir: Path,
) -> Dict[str, np.ndarray]:
    ensure_dir(outdir)

    lattice, hilbert, ham = make_lattice_and_hamiltonian(cfg.L, cfg.J1, cfg.J2)
    n_sites = lattice.n_nodes
    sampler = build_sampler(hilbert, lattice, cfg.d_max)

    ans = ansatz.lower()
    if ans == "mlp":
        model = build_mlp_model_two_layers(n_sites=n_sites, hidden_scale=cfg.mlp_hidden_scale, param_dtype=cfg.param_dtype)
        lr = cfg.mlp_lr
    elif ans == "rbm":
        model = build_rbm_model(alpha=cfg.rbm_alpha, param_dtype=cfg.param_dtype)
        lr = cfg.rbm_lr
    else:
        raise ValueError("ansatz must be 'MLP' or 'RBM'")

    opt = build_optimizer(opt_name, lr=lr, weight_decay=cfg.weight_decay)

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

    # History CSV with metadata lines
    csv_path = outdir / f"{ansatz}_{dtype_label}_{opt_name}_history.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(f"# {ansatz} optimizer sweep | dtype={dtype_label} | optimizer={opt_name}\n")
        f.write(f"# L={cfg.L} (N_sites={n_sites}) | J1={cfg.J1} | J2={cfg.J2}\n")
        f.write(f"# n_samples={cfg.n_samples} | n_iter={cfg.n_iter} | diag_shift={cfg.diag_shift} | seed={cfg.seed}\n")
        if opt_name.lower() == "adamw":
            f.write(f"# AdamW weight_decay={cfg.weight_decay}\n")
        f.write("iter,energy_mean,energy_sigma,energy_per_site_mean,energy_per_site_sigma\n")
        for i, em, es, eps, epe in zip(iters, E_mean, E_sigma, e_site, e_site_err):
            f.write(f"{int(i)},{em:.12f},{es:.12f},{eps:.12f},{epe:.12f}\n")

    meta = {
        "ansatz": ansatz,
        "dtype_label": dtype_label,
        "optimizer": opt_name,
        "L": cfg.L,
        "n_sites": int(n_sites),
        "J1": cfg.J1,
        "J2": cfg.J2,
        "n_samples": cfg.n_samples,
        "n_discard_per_chain": cfg.n_discard_per_chain,
        "n_iter": cfg.n_iter,
        "diag_shift": cfg.diag_shift,
        "seed": cfg.seed,
        "param_dtype": str(cfg.param_dtype),
        "learning_rate": lr,
        "weight_decay": (cfg.weight_decay if opt_name.lower() == "adamw" else 0.0),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "n_parameters": int(vstate.n_parameters),
        "runtime_seconds": float(t1 - t0),
        "final_energy_per_site": float(e_site[-1]),
        "final_energy_per_site_err": float(e_site_err[-1]),
    }
    save_json(outdir / "run_meta.json", meta)

    return {
        "iters": iters,
        "e_site": e_site,
        "e_site_err": e_site_err,
        "n_sites": np.array([n_sites], dtype=np.int64),
        "runtime_s": np.array([t1 - t0], dtype=np.float64),
        "n_params": np.array([int(vstate.n_parameters)], dtype=np.int64),
    }


# -----------------------------
# Summaries
# -----------------------------
def write_per_run_summary(outdir: Path, rows: List[Dict[str, Any]]):
    ensure_dir(outdir)

    csv_path = outdir / "per_run_summary.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("# One row per run: (ansatz, dtype, optimizer)\n")
        f.write("ansatz,dtype,optimizer,n_params,runtime_s,final_e_site,final_e_site_err,paper_true_e_site,ed_true_e_site\n")
        for r in rows:
            f.write(
                f"{r['ansatz']},{r['dtype']},{r['optimizer']},"
                f"{r['n_params']},{r['runtime_s']:.6f},"
                f"{r['final_e_site']:.12f},{r['final_e_site_err']:.12f},"
                f"{'' if r['paper_true'] is None else f'{r['paper_true']:.6f}'},"
                f"{'' if r['ed_true'] is None else f'{r['ed_true']:.12f}'}\n"
            )

    headers = ["ANSATZ", "DTYPE", "OPT", "N_PARAMS", "RUNTIME(s)", "FINAL E/SITE", "ERR", "PAPER", "ED"]
    table = []
    for r in rows:
        table.append([
            r["ansatz"],
            r["dtype"],
            r["optimizer"],
            str(r["n_params"]),
            f"{r['runtime_s']:.2f}",
            f"{r['final_e_site']:.8f}",
            f"{r['final_e_site_err']:.8f}",
            ("" if r["paper_true"] is None else f"{r['paper_true']:.5f}"),
            ("" if r["ed_true"] is None else f"{r['ed_true']:.8f}"),
        ])

    colw = [len(h) for h in headers]
    for row in table:
        for i, cell in enumerate(row):
            colw[i] = max(colw[i], len(cell))

    def line(ch="-"):
        return "+" + "+".join([ch * (w + 2) for w in colw]) + "+\n"

    txt_path = outdir / "per_run_summary.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(line("-"))
        f.write("| " + " | ".join([headers[i].ljust(colw[i]) for i in range(len(headers))]) + " |\n")
        f.write(line("="))
        for row in table:
            f.write("| " + " | ".join([row[i].ljust(colw[i]) for i in range(len(headers))]) + " |\n")
        f.write(line("-"))


def write_overall_results(outdir: Path, rows: List[Dict[str, Any]]):
    """
    One row per (ansatz,dtype): Adam vs AdamW side-by-side.
    """
    ensure_dir(outdir)

    csv_path = outdir / "overall_results.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("# One row per (ansatz,dtype): Adam vs AdamW side-by-side\n")
        f.write("ansatz,dtype,paper_true,ed_true,adam_e,adam_err,adamw_e,adamw_err,d_adam_paper,d_adamw_paper,d_adam_ed,d_adamw_ed\n")
        for r in rows:
            def s(x, prec=12):
                return "" if x is None else f"{x:.{prec}f}"
            f.write(
                f"{r['ansatz']},{r['dtype']},"
                f"{'' if r['paper_true'] is None else f'{r['paper_true']:.6f}'},"
                f"{s(r['ed_true'])},"
                f"{s(r['adam_e'])},{s(r['adam_err'])},"
                f"{s(r['adamw_e'])},{s(r['adamw_err'])},"
                f"{s(r['d_adam_paper'])},{s(r['d_adamw_paper'])},"
                f"{s(r['d_adam_ed'])},{s(r['d_adamw_ed'])}\n"
            )

    headers = ["ANSATZ", "DTYPE", "PAPER", "ED", "ADAM E", "±", "ADAMW E", "±", "ΔADAM(P)", "ΔADAMW(P)", "ΔADAM(ED)", "ΔADAMW(ED)"]
    def fmt(x, prec=8):
        return "" if x is None else f"{x:.{prec}f}"

    table = []
    for r in rows:
        table.append([
            r["ansatz"],
            r["dtype"],
            ("" if r["paper_true"] is None else f"{r['paper_true']:.5f}"),
            fmt(r["ed_true"], 8),
            fmt(r["adam_e"], 8),
            fmt(r["adam_err"], 8),
            fmt(r["adamw_e"], 8),
            fmt(r["adamw_err"], 8),
            fmt(r["d_adam_paper"], 8),
            fmt(r["d_adamw_paper"], 8),
            fmt(r["d_adam_ed"], 8),
            fmt(r["d_adamw_ed"], 8),
        ])

    colw = [len(h) for h in headers]
    for row in table:
        for i, cell in enumerate(row):
            colw[i] = max(colw[i], len(cell))

    def line(ch="-"):
        return "+" + "+".join([ch * (w + 2) for w in colw]) + "+\n"

    txt_path = outdir / "overall_results.txt"
    with txt_path.open("w", encoding="utf-8") as f:
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
        description="Compare Adam vs AdamW for MLP(2 layers, log_cosh) and RBM, real vs complex, at J2=0.5."
    )

    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--J1", type=float, default=1.0)
    parser.add_argument("--J2", type=float, default=0.5)

    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_iter", type=int, default=600)
    parser.add_argument("--discard", type=int, default=50)
    parser.add_argument("--diag_shift", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--mlp_hidden_scale", type=int, default=1)

    parser.add_argument("--rbm_lr", type=float, default=1e-2)
    parser.add_argument("--rbm_alpha", type=int, default=4)

    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for AdamW (ignored for Adam).")

    parser.add_argument("--d_max", type=int, default=2)

    parser.add_argument("--platform", type=str, default=_platform,
                        choices=["auto", "cpu", "gpu", "tpu"])
    parser.add_argument("--out", type=str, default="results_optimizer_adam_vs_adamw_J2_0p5")

    parser.add_argument("--ed_max_sites", type=int, default=20,
                        help="Compute NetKet ED reference only if N_sites <= ed_max_sites.")

    args = parser.parse_args()

    out_root = Path(args.out)
    ensure_dir(out_root)

    n_sites = args.L * args.L
    paper_true = PAPER_TRUE_E_SITE if abs(args.J2 - 0.5) < 1e-12 else None
    ed_true = maybe_exact_ground_state_energy_per_site(args.L, args.J1, args.J2, args.ed_max_sites)

    print("\n===================================================")
    print("Optimizer Sweep: Adam vs AdamW (VMC_SR)")
    print("===================================================")
    print(f"Requested platform: {args.platform}")
    print(f"JAX backend:        {jax.default_backend()}")
    print("JAX devices:        " + ", ".join([str(d) for d in jax.devices()]))
    print("---------------------------------------------------")
    print(f"L={args.L} -> N_sites={n_sites}")
    print(f"J1={args.J1} | J2={args.J2}")
    print("---------------------------------------------------")
    print(f"n_samples={args.n_samples} | n_iter={args.n_iter} | discard={args.discard} | diag_shift={args.diag_shift} | seed={args.seed}")
    print(f"MLP: lr={args.mlp_lr} | hidden_dims=(N,N) with N=n_sites*{args.mlp_hidden_scale} = {n_sites*args.mlp_hidden_scale} | act=log_cosh")
    print(f"RBM: lr={args.rbm_lr} | alpha={args.rbm_alpha}")
    print(f"AdamW weight_decay={args.weight_decay}")
    if paper_true is not None:
        print(f"Paper reference E/site: {paper_true:.6f}")
    else:
        print("Paper reference E/site: (not provided for this J2)")
    if ed_true is None:
        print(f"NetKet ED reference:    (skipped; requires N_sites <= {args.ed_max_sites})")
    else:
        print(f"NetKet ED reference:    {ed_true:.12f}")
    print("===================================================\n")

    # Save config
    save_json(out_root / "sweep_config.json", {
        "L": args.L,
        "J1": args.J1,
        "J2": args.J2,
        "n_samples": args.n_samples,
        "n_iter": args.n_iter,
        "discard": args.discard,
        "diag_shift": args.diag_shift,
        "seed": args.seed,
        "mlp": {"lr": args.mlp_lr, "hidden_scale": args.mlp_hidden_scale, "activation": "log_cosh"},
        "rbm": {"lr": args.rbm_lr, "alpha": args.rbm_alpha},
        "weight_decay": args.weight_decay,
        "paper_true_e_site": paper_true,
        "ed_true_e_site": ed_true,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
    })

    # Output directories
    buckets = {
        ("MLP", "real", "Adam"):  out_root / "MLP_real" / "Adam",
        ("MLP", "real", "AdamW"): out_root / "MLP_real" / "AdamW",
        ("MLP", "complex", "Adam"):  out_root / "MLP_complex" / "Adam",
        ("MLP", "complex", "AdamW"): out_root / "MLP_complex" / "AdamW",
        ("RBM", "real", "Adam"):  out_root / "RBM_real" / "Adam",
        ("RBM", "real", "AdamW"): out_root / "RBM_real" / "AdamW",
        ("RBM", "complex", "Adam"):  out_root / "RBM_complex" / "Adam",
        ("RBM", "complex", "AdamW"): out_root / "RBM_complex" / "AdamW",
    }
    for p in buckets.values():
        ensure_dir(p)
    compare_dir = out_root / "compare"
    ensure_dir(compare_dir)

    # Store histories
    # hists[(ansatz, dtype, opt)] = hist
    hists: Dict[Tuple[str, str, str], Dict[str, np.ndarray]] = {}

    per_run_rows: List[Dict[str, Any]] = []

    for ansatz in ["MLP", "RBM"]:
        for dtype_label, dtype in [("real", jnp.float64), ("complex", jnp.complex128)]:
            for opt_name in ["Adam", "AdamW"]:
                cfg = RunConfig(
                    L=args.L, J1=args.J1, J2=args.J2,
                    n_samples=args.n_samples,
                    n_discard_per_chain=args.discard,
                    n_iter=args.n_iter,
                    diag_shift=args.diag_shift,
                    seed=args.seed,
                    d_max=args.d_max,
                    mlp_lr=args.mlp_lr,
                    mlp_hidden_scale=args.mlp_hidden_scale,
                    rbm_lr=args.rbm_lr,
                    rbm_alpha=args.rbm_alpha,
                    weight_decay=args.weight_decay,
                    param_dtype=dtype,
                )

                outdir = buckets[(ansatz, dtype_label, opt_name)]
                print(f"[RUN] {ansatz} | dtype={dtype_label} | opt={opt_name} -> {outdir}")
                hist = run_single(ansatz, dtype_label, opt_name, cfg, outdir)
                hists[(ansatz, dtype_label, opt_name)] = hist

                # Per-run plot (3 variants)
                title_base = (
                    f"{ansatz} | {opt_name} | dtype={dtype_label} | "
                    f"L={args.L} (N_sites={n_sites}) | J1={args.J1} | J2={args.J2} | samples={args.n_samples}"
                )
                plot_three_variants_energy_curve(
                    outdir=outdir,
                    base_name=f"{ansatz}_{dtype_label}_{opt_name}_energy_per_site",
                    title_base=title_base,
                    iters=hist["iters"],
                    curves=[(f"{opt_name}", hist["e_site"], hist["e_site_err"])],
                    paper_true=paper_true,
                    ed_true=ed_true,
                )

                final_e = float(hist["e_site"][-1])
                final_err = float(hist["e_site_err"][-1])

                print(f"      Final E/site: {fmt_pm(final_e, final_err)}")
                if paper_true is not None:
                    print(f"      Paper true:   {paper_true:.6f}  (Δ={final_e - paper_true:+.6f})")
                if ed_true is not None:
                    print(f"      NetKet ED:    {ed_true:.12f}  (Δ={final_e - ed_true:+.6f})")
                print("")

                per_run_rows.append({
                    "ansatz": ansatz,
                    "dtype": dtype_label,
                    "optimizer": opt_name,
                    "n_params": int(hist["n_params"][0]),
                    "runtime_s": float(hist["runtime_s"][0]),
                    "final_e_site": final_e,
                    "final_e_site_err": final_err,
                    "paper_true": paper_true,
                    "ed_true": ed_true,
                })

    # Write per-run summaries
    write_per_run_summary(out_root, per_run_rows)

    # -----------------------------------------
    # Comparisons per (ansatz,dtype): Adam vs AdamW
    # -----------------------------------------
    overall_rows: List[Dict[str, Any]] = []

    def delta(a: float, b: Optional[float]) -> Optional[float]:
        return None if b is None else a - float(b)

    for ansatz in ["MLP", "RBM"]:
        for dtype_label in ["real", "complex"]:
            h_adam = hists[(ansatz, dtype_label, "Adam")]
            h_adamw = hists[(ansatz, dtype_label, "AdamW")]

            # comparison plot (3 variants)
            base = f"compare_{ansatz}_{dtype_label}_Adam_vs_AdamW"
            title_base = (
                f"{ansatz} | Adam vs AdamW | dtype={dtype_label} | "
                f"L={args.L} (N_sites={n_sites}) | J1={args.J1} | J2={args.J2} | samples={args.n_samples}"
            )
            plot_three_variants_energy_curve(
                outdir=compare_dir,
                base_name=base,
                title_base=title_base,
                iters=h_adam["iters"],
                curves=[
                    ("Adam",  h_adam["e_site"],  h_adam["e_site_err"]),
                    ("AdamW", h_adamw["e_site"], h_adamw["e_site_err"]),
                ],
                paper_true=paper_true,
                ed_true=ed_true,
            )

            adam_e = float(h_adam["e_site"][-1]); adam_err = float(h_adam["e_site_err"][-1])
            adamw_e = float(h_adamw["e_site"][-1]); adamw_err = float(h_adamw["e_site_err"][-1])

            overall_rows.append({
                "ansatz": ansatz,
                "dtype": dtype_label,
                "paper_true": paper_true,
                "ed_true": ed_true,
                "adam_e": adam_e, "adam_err": adam_err,
                "adamw_e": adamw_e, "adamw_err": adamw_err,
                "d_adam_paper": delta(adam_e, paper_true),
                "d_adamw_paper": delta(adamw_e, paper_true),
                "d_adam_ed": delta(adam_e, ed_true),
                "d_adamw_ed": delta(adamw_e, ed_true),
            })

    write_overall_results(out_root, overall_rows)

    # -----------------------------------------
    # One plot: all 8 runs together (3 variants)
    # -----------------------------------------
    curves_all = []
    it_ref = hists[("MLP", "real", "Adam")]["iters"]
    for ansatz in ["MLP", "RBM"]:
        for dtype_label in ["real", "complex"]:
            for opt_name in ["Adam", "AdamW"]:
                h = hists[(ansatz, dtype_label, opt_name)]
                curves_all.append((f"{ansatz}-{dtype_label}-{opt_name}", h["e_site"], h["e_site_err"]))

    plot_three_variants_energy_curve(
        outdir=compare_dir,
        base_name="compare_all_8_runs",
        title_base=(
            f"All runs | Adam vs AdamW | MLP+RBM | real+complex | "
            f"L={args.L} (N_sites={n_sites}) | J1={args.J1} | J2={args.J2} | samples={args.n_samples}"
        ),
        iters=it_ref,
        curves=curves_all,
        paper_true=paper_true,
        ed_true=ed_true,
    )

    print("\n===================================================")
    print("DONE")
    print("===================================================")
    print(f"Outputs saved to: {out_root.resolve()}")
    print("Summary files:")
    print("  - per_run_summary.csv / per_run_summary.txt   (one row per run)")
    print("  - overall_results.csv / overall_results.txt   (Adam vs AdamW side-by-side per (ansatz,dtype))")
    print("Compare plots in: compare/")
    print("Note: NetKet ED reference is skipped automatically unless N_sites <= ed_max_sites.\n")


if __name__ == "__main__":
    main()
