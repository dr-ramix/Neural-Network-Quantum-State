#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Activation-function sweep for MLP Neural Quantum States (NetKet) on J1-J2 Heisenberg model:

Fixed:
  - J1 = 1.0
  - J2 in [0.5, 0.6]
  - LxL square lattice (PBC), total_sz = 0 sector
  - MLP only with TWO hidden layers (N, N) where N = n_sites * hidden_scale
  - Run each activation once with REAL params, once with COMPLEX params

Key requirement:
  Some activations are not holomorphic / not defined for complex inputs (ReLU, GELU, SiLU).
  For COMPLEX parameters we therefore apply the activation to real/imag parts separately:
     f_c(z) = f(Re(z)) + i f(Im(z))

Usage:
  python mlp_activation_sweep_real_vs_complex.py --platform gpu
  python mlp_activation_sweep_real_vs_complex.py --platform cpu --L 4 --ed_max_sites 16
"""

import os
import json
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple

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

# ESSENTIAL: Enable 64-bit precision for physics simulations
jax.config.update("jax_enable_x64", True)

import netket as nk

# Import log_cosh handling different NetKet versions
try:
    from netket.nn import log_cosh
except ImportError:
    try:
        from netket.nn.activation import log_cosh
    except ImportError:
        # Fallback implementation
        def log_cosh(x):
            return jnp.log(jnp.cosh(x))


# -----------------------------
# Paper true values (energy per site)
# -----------------------------
PAPER_TRUE_E_SITE = {
    0.5: -0.50381,
    0.6: -0.49518,
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
    # max_neighbor_order=2 ensures edges for NN (color 0) and NNN (color 1) are generated
    lattice = nk.graph.Square(length=L, max_neighbor_order=2, pbc=True)
    hilbert = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=lattice.n_nodes)
    
    # NetKet Heisenberg applies J[0] to edges with color 0, J[1] to color 1
    ham = nk.operator.Heisenberg(
        hilbert=hilbert,
        graph=lattice,
        J=[J1, J2],
        sign_rule=[False, False], 
    )
    return lattice, hilbert, ham

def build_sampler(hilbert, lattice, d_max: int = 2):
    return nk.sampler.MetropolisExchange(hilbert=hilbert, graph=lattice, d_max=d_max)

def build_vmc_sr_driver(hamiltonian, vstate, optimizer, diag_shift: float):
    """
    Setup VMC driver with Stochastic Reconfiguration (SR).
    """
    # Modern NetKet approach (v3.0+)
    if hasattr(nk.optimizer, "SR"):
        sr = nk.optimizer.SR(diag_shift=diag_shift)
        return nk.driver.VMC(
            hamiltonian=hamiltonian,
            optimizer=optimizer,
            variational_state=vstate,
            preconditioner=sr,
        )
        
    # Older NetKet approach fallback
    if hasattr(nk.driver, "VMC_SR"):
        return nk.driver.VMC_SR(
            hamiltonian=hamiltonian,
            optimizer=optimizer,
            diag_shift=diag_shift,
            variational_state=vstate,
        )

    raise RuntimeError("Could not configure VMC with SR. Check NetKet version.")


# -----------------------------
# ED reference (NetKet) if feasible
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
        # NetKet v3+ lanczos_ed
        if hasattr(nk.exact, "lanczos_ed"):
            res = nk.exact.lanczos_ed(ham, k=1, compute_eigenvectors=False)
            # Handle different return types across versions
            if hasattr(res, "eigenvalues"):
                e0 = float(np.asarray(res.eigenvalues)[0])
            elif isinstance(res, dict) and "eigenvalues" in res:
                e0 = float(np.asarray(res["eigenvalues"])[0])
            else:
                # Some versions return just the array or list
                arr = np.asarray(res)
                e0 = float(arr[0] if arr.size > 0 else arr)
            return e0 / n_sites

        # Fallback to dense diagonalization
        if hasattr(nk.exact, "diag"):
            evals = nk.exact.diag(ham)
            e0 = float(np.min(np.asarray(evals)))
            return e0 / n_sites

    except Exception as e:
        print(f"Warning: ED failed: {e}")
        return None

    return None


# -----------------------------
# Activation functions
# -----------------------------
def complexify_activation(act_real: Callable) -> Callable:
    """
    Make an activation usable on complex inputs by applying it to Re and Im separately:
        f_c(z) = f(Re z) + i f(Im z)
    """
    def act(z):
        # jnp.iscomplexobj is correct for checking dtype of JAX tracers
        if jnp.iscomplexobj(z):
            return act_real(jnp.real(z)) + 1j * act_real(jnp.imag(z))
        return act_real(z)
    return act


def get_activation_map() -> Dict[str, Callable]:
    return {
        "relu": jax.nn.relu,
        "silu": jax.nn.silu,
        "gelu": jax.nn.gelu,
        "log_cosh": log_cosh,
    }


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
    n_iter: int = 600
    diag_shift: float = 0.01
    seed: int = 1234

    d_max: int = 2

    # MLP (N,N)
    mlp_lr: float = 1e-3
    hidden_scale: int = 1 

    param_dtype: Any = jnp.float64


def build_mlp_model_two_layers(
    n_sites: int,
    hidden_scale: int,
    param_dtype: Any,
    activation: Callable,
):
    h = int(n_sites * hidden_scale)
    # NetKet MLP automatically creates dense layers with the given activation
    return nk.models.MLP(
        hidden_dims=(h, h),
        param_dtype=param_dtype,
        hidden_activations=activation,
        output_activation=None,
        use_output_bias=True,
    )


# -----------------------------
# Plotting helpers
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
        ax.axhline(tval, linestyle="--", linewidth=2.0, color='black', label=tlab)

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
# Run a single experiment
# -----------------------------
def run_single_mlp_activation(
    cfg: RunConfig,
    dtype_label: str,
    act_name: str,
    act_fn: Callable,
    outdir: Path,
) -> Dict[str, np.ndarray]:
    ensure_dir(outdir)

    lattice, hilbert, ham = make_lattice_and_hamiltonian(cfg.L, cfg.J1, cfg.J2)
    n_sites = lattice.n_nodes

    sampler = build_sampler(hilbert, lattice, cfg.d_max)
    
    model = build_mlp_model_two_layers(
        n_sites=n_sites,
        hidden_scale=cfg.hidden_scale,
        param_dtype=cfg.param_dtype,
        activation=act_fn,
    )
    
    # NetKet 3 optimizer is an optax wrapper
    opt = nk.optimizer.Adam(learning_rate=cfg.mlp_lr)

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

    # Data extraction - Handling NetKet 3+ dictionary structure
    data = log.data
    
    # Check structure to ensure compatibility
    if "Energy" not in data:
         raise RuntimeError("Energy not found in log data.")
         
    e_log = data["Energy"]
    
    # NetKet 3 returns a dict for stats, NetKet 2 returned an object
    if isinstance(e_log, dict):
        iters = np.asarray(e_log["iters"], dtype=int)
        E_mean = np.asarray(e_log["Mean"], dtype=float) # .real handled by np.asarray usually, but explicit casting is safer
        E_sigma = np.asarray(e_log["Sigma"], dtype=float)
        # Handle complex numbers if they leaked (should be real for Energy)
        if np.iscomplexobj(E_mean): E_mean = E_mean.real
        if np.iscomplexobj(E_sigma): E_sigma = E_sigma.real
    else:
        # Fallback for object-style logs
        iters = np.asarray(e_log.iters, dtype=int)
        E_mean = np.asarray(e_log.Mean.real, dtype=float)
        E_sigma = np.asarray(e_log.Sigma.real, dtype=float)

    e_site = E_mean / n_sites
    e_site_err = E_sigma / n_sites

    # History CSV with metadata lines
    csv_path = outdir / f"mlp_{dtype_label}_{act_name}_J2_{cfg.J2:.2f}_history.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(f"# MLP (N,N) activation sweep | dtype={dtype_label} | activation={act_name}\n")
        f.write(f"# L={cfg.L} (N_sites={n_sites}) | J1={cfg.J1} | J2={cfg.J2}\n")
        f.write(f"# n_samples={cfg.n_samples} | n_iter={cfg.n_iter} | diag_shift={cfg.diag_shift} | seed={cfg.seed}\n")
        f.write("iter,energy_mean,energy_sigma,energy_per_site_mean,energy_per_site_sigma\n")
        for i, em, es, eps, epe in zip(iters, E_mean, E_sigma, e_site, e_site_err):
            f.write(f"{int(i)},{em:.12f},{es:.12f},{eps:.12f},{epe:.12f}\n")

    meta = {
        "arch": "MLP",
        "hidden_layers": 2,
        "hidden_dims": "N,N (N=n_sites*hidden_scale)",
        "activation": act_name,
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
        "param_dtype": str(cfg.param_dtype),
        "optimizer": {"type": "Adam", "learning_rate": cfg.mlp_lr},
        "hidden_scale": cfg.hidden_scale,
        "n_parameters": int(vstate.n_parameters),
        "runtime_seconds": float(t1 - t0),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
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
        f.write("# Per-run summary (one row per dtype, activation, J2)\n")
        f.write("dtype,activation,J2,n_params,runtime_s,final_e_site,final_e_site_err,paper_true_e_site,ed_true_e_site\n")
        for r in rows:
            f.write(
                f"{r['dtype']},{r['activation']},{r['J2']:.2f},"
                f"{r['n_params']},{r['runtime_s']:.6f},"
                f"{r['final_e_site']:.12f},{r['final_e_site_err']:.12f},"
                f"{'' if r['paper_true'] is None else f'{r['paper_true']:.6f}'},"
                f"{'' if r['ed_true'] is None else f'{r['ed_true']:.12f}'}\n"
            )

    headers = ["DTYPE", "ACT", "J2", "N_PARAMS", "RUNTIME(s)", "FINAL E/SITE", "ERR", "PAPER", "ED"]
    table = []
    for r in rows:
        table.append([
            r["dtype"],
            r["activation"],
            f"{r['J2']:.2f}",
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
    ensure_dir(outdir)

    csv_path = outdir / "overall_results.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("# Overall results: one row per (activation,J2), real vs complex side-by-side\n")
        f.write("activation,J2,paper_true,ed_true,real_e,real_err,complex_e,complex_err,d_real_paper,d_complex_paper,d_real_ed,d_complex_ed\n")
        for r in rows:
            def s(x, prec=12):
                return "" if x is None else f"{x:.{prec}f}"
            f.write(
                f"{r['activation']},{r['J2']:.2f},"
                f"{'' if r['paper_true'] is None else f'{r['paper_true']:.6f}'},"
                f"{s(r['ed_true'])},"
                f"{s(r['real_e'])},{s(r['real_err'])},"
                f"{s(r['complex_e'])},{s(r['complex_err'])},"
                f"{s(r['d_real_paper'])},{s(r['d_complex_paper'])},"
                f"{s(r['d_real_ed'])},{s(r['d_complex_ed'])}\n"
            )

    headers = ["ACT", "J2", "PAPER", "ED", "REAL E", "±", "CPLX E", "±", "ΔREAL(P)", "ΔCPLX(P)", "ΔREAL(ED)", "ΔCPLX(ED)"]
    def fmt(x, prec=8):
        return "" if x is None else f"{x:.{prec}f}"

    table = []
    for r in rows:
        table.append([
            r["activation"],
            f"{r['J2']:.2f}",
            ("" if r["paper_true"] is None else f"{r['paper_true']:.5f}"),
            fmt(r["ed_true"], 8),
            fmt(r["real_e"], 8),
            fmt(r["real_err"], 8),
            fmt(r["complex_e"], 8),
            fmt(r["complex_err"], 8),
            fmt(r["d_real_paper"], 8),
            fmt(r["d_complex_paper"], 8),
            fmt(r["d_real_ed"], 8),
            fmt(r["d_complex_ed"], 8),
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
        description="Activation sweep for MLP (two hidden layers) with real vs complex params on J1-J2 model."
    )
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--J1", type=float, default=1.0)
    parser.add_argument("--J2_list", type=str, default="0.5,0.6")

    parser.add_argument("--activations", type=str, default="relu,silu,gelu,log_cosh",
                        help="Comma-separated activation names from {relu,silu,gelu,log_cosh}.")
    parser.add_argument("--complex_activation_mode", type=str, default="split",
                        choices=["split", "native"],
                        help=(
                            "How to handle activations for complex runs. "
                            "'split' applies f(Re)+i f(Im) for non-complex activations; "
                            "'native' uses the same function directly on complex inputs (may fail for relu/gelu/silu)."
                        ))

    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_iter", type=int, default=600)
    parser.add_argument("--discard", type=int, default=50)
    parser.add_argument("--diag_shift", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--hidden_scale", type=int, default=1)
    parser.add_argument("--d_max", type=int, default=2)

    parser.add_argument("--platform", type=str, default=_platform,
                        choices=["auto", "cpu", "gpu", "tpu"])
    parser.add_argument("--out", type=str, default="results_mlp_activation_sweep_real_vs_complex")

    parser.add_argument("--ed_max_sites", type=int, default=20,
                        help="Compute NetKet ED reference only if N_sites <= ed_max_sites.")

    args = parser.parse_args()

    # Parse lists
    J2_list = [float(x.strip()) for x in args.J2_list.split(",") if x.strip()]
    act_names = [x.strip() for x in args.activations.split(",") if x.strip()]

    act_map = get_activation_map()
    for a in act_names:
        if a not in act_map:
            raise ValueError(f"Unknown activation '{a}'. Allowed: {list(act_map.keys())}")

    out_root = Path(args.out)
    ensure_dir(out_root)

    n_sites = args.L * args.L

    print("\n===================================================")
    print("MLP Activation Sweep (two hidden layers: N,N)")
    print("===================================================")
    print(f"Requested platform: {args.platform}")
    print(f"JAX backend:        {jax.default_backend()}")
    print("JAX devices:        " + ", ".join([str(d) for d in jax.devices()]))
    print("---------------------------------------------------")
    print(f"L={args.L} -> N_sites={n_sites}")
    print(f"J1={args.J1} | J2_list={J2_list}")
    print(f"Hidden dims: (N,N) with N = n_sites*hidden_scale = {n_sites}*{args.hidden_scale} = {n_sites*args.hidden_scale}")
    print(f"Activations: {act_names}")
    print(f"Complex activation mode: {args.complex_activation_mode}")
    print("---------------------------------------------------")
    print(f"n_samples={args.n_samples} | n_iter={args.n_iter} | discard={args.discard} | diag_shift={args.diag_shift} | seed={args.seed}")
    print(f"MLP Adam lr={args.mlp_lr} | sampler=MetropolisExchange(d_max={args.d_max})")
    print(f"ED reference: enabled only if N_sites <= {args.ed_max_sites}")
    print("===================================================\n")

    # Save config
    save_json(out_root / "sweep_config.json", {
        "L": args.L,
        "J1": args.J1,
        "J2_list": J2_list,
        "activations": act_names,
        "hidden_layers": 2,
        "hidden_scale": args.hidden_scale,
        "n_samples": args.n_samples,
        "n_iter": args.n_iter,
        "discard": args.discard,
        "diag_shift": args.diag_shift,
        "seed": args.seed,
        "mlp_lr": args.mlp_lr,
        "d_max": args.d_max,
        "complex_activation_mode": args.complex_activation_mode,
        "paper_true_values": {str(k): v for k, v in PAPER_TRUE_E_SITE.items()},
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
    })

    # Output folders
    real_root = out_root / "MLP_activations_real"
    cplx_root = out_root / "MLP_activations_complex"
    cmp_root = out_root / "compare_per_j2"
    for p in [real_root, cplx_root, cmp_root]:
        ensure_dir(p)

    # results[(dtype, J2, act)] = hist
    results: Dict[Tuple[str, float, str], Dict[str, np.ndarray]] = {}

    per_run_rows: List[Dict[str, Any]] = []
    overall_rows: List[Dict[str, Any]] = []

    for J2 in J2_list:
        paper_true = PAPER_TRUE_E_SITE.get(J2, None)
        ed_true = maybe_exact_ground_state_energy_per_site(args.L, args.J1, J2, args.ed_max_sites)

        print(f"=== J2 = {J2:.2f} ===")
        print(f"Paper reference E/site: {paper_true if paper_true is not None else 'N/A'}")
        if ed_true is None:
            print("NetKet ED reference:    (skipped / not feasible for this size)")
        else:
            print(f"NetKet ED reference:    {ed_true:.12f}")
        print("")

        for dtype_label, dtype in [("real", jnp.float64), ("complex", jnp.complex128)]:
            print(f"--- dtype = {dtype_label.upper()} ---")

            for act_name in act_names:
                base_act = act_map[act_name]

                if dtype_label == "complex":
                    if args.complex_activation_mode == "split":
                        # For non-holomorphic activations, do f(Re)+i f(Im)
                        # For log_cosh (analytic) we allow native complex by default.
                        if act_name == "log_cosh":
                            act_fn = base_act
                            act_note = "native complex (analytic)"
                        else:
                            act_fn = complexify_activation(base_act)
                            act_note = "split: f(Re)+i f(Im)"
                    else:
                        act_fn = base_act
                        act_note = "native (may fail for relu/silu/gelu)"
                else:
                    act_fn = base_act
                    act_note = "real"

                cfg = RunConfig(
                    L=args.L, J1=args.J1, J2=J2,
                    n_samples=args.n_samples,
                    n_discard_per_chain=args.discard,
                    n_iter=args.n_iter,
                    diag_shift=args.diag_shift,
                    seed=args.seed,
                    d_max=args.d_max,
                    mlp_lr=args.mlp_lr,
                    hidden_scale=args.hidden_scale,
                    param_dtype=dtype,
                )

                outdir = (real_root if dtype_label == "real" else cplx_root) / f"J2_{J2:.2f}" / f"act_{act_name}"
                tag = f"MLP_{dtype_label}_{act_name}_J2_{J2:.2f}"

                print(f"[RUN] {tag}  | activation_mode: {act_note}")
                hist = run_single_mlp_activation(cfg, dtype_label=dtype_label, act_name=act_name, act_fn=act_fn, outdir=outdir)
                results[(dtype_label, J2, act_name)] = hist

                final_e = float(hist["e_site"][-1])
                final_err = float(hist["e_site_err"][-1])

                print(f"      Final E/site: {fmt_pm(final_e, final_err)}")
                if paper_true is not None:
                    print(f"      Paper true:   {paper_true:.6f}  (Δ={final_e - paper_true:+.6f})")
                if ed_true is not None:
                    print(f"      NetKet ED:    {ed_true:.12f}  (Δ={final_e - ed_true:+.6f})")
                print("")

                # Per-run plots (3 variants)
                title_base = (
                    f"MLP (N,N) | act={act_name} | dtype={dtype_label} | hidden_scale={args.hidden_scale} | "
                    f"L={args.L} (N_sites={n_sites}) | J1={args.J1} | J2={J2} | samples={args.n_samples}"
                )
                plot_three_variants_energy_curve(
                    outdir=outdir,
                    base_name=f"{tag}_energy_per_site",
                    title_base=title_base,
                    iters=hist["iters"],
                    curves=[(f"{act_name}", hist["e_site"], hist["e_site_err"])],
                    paper_true=paper_true,
                    ed_true=ed_true,
                )

                per_run_rows.append({
                    "dtype": dtype_label,
                    "activation": act_name,
                    "J2": J2,
                    "n_params": int(hist["n_params"][0]),
                    "runtime_s": float(hist["runtime_s"][0]),
                    "final_e_site": final_e,
                    "final_e_site_err": final_err,
                    "paper_true": paper_true,
                    "ed_true": ed_true,
                })

        # -------------------------------------------------------
        # Per-J2 comparisons:
        #   (1) compare activations (REAL) on one plot (3 variants)
        #   (2) compare activations (COMPLEX) on one plot (3 variants)
        #   (3) for each activation: REAL vs COMPLEX (3 variants each)
        #   (4) all curves (real+complex across activations) (3 variants)
        # -------------------------------------------------------
        cmp_j2_dir = cmp_root / f"J2_{J2:.2f}"
        ensure_dir(cmp_j2_dir)

        # (1) real: activations comparison
        curves_real = []
        it_ref = None
        for act_name in act_names:
            h = results[("real", J2, act_name)]
            if it_ref is None:
                it_ref = h["iters"]
            curves_real.append((act_name, h["e_site"], h["e_site_err"]))

        plot_three_variants_energy_curve(
            outdir=cmp_j2_dir,
            base_name=f"compare_activations_real_J2_{J2:.2f}",
            title_base=f"Activation comparison (REAL params) | MLP (N,N) | L={args.L} | J1={args.J1} | J2={J2} | samples={args.n_samples}",
            iters=it_ref,
            curves=curves_real,
            paper_true=paper_true,
            ed_true=ed_true,
        )

        # (2) complex: activations comparison
        curves_cplx = []
        it_ref = None
        for act_name in act_names:
            h = results[("complex", J2, act_name)]
            if it_ref is None:
                it_ref = h["iters"]
            curves_cplx.append((act_name, h["e_site"], h["e_site_err"]))

        plot_three_variants_energy_curve(
            outdir=cmp_j2_dir,
            base_name=f"compare_activations_complex_J2_{J2:.2f}",
            title_base=f"Activation comparison (COMPLEX params) | MLP (N,N) | L={args.L} | J1={args.J1} | J2={J2} | samples={args.n_samples}",
            iters=it_ref,
            curves=curves_cplx,
            paper_true=paper_true,
            ed_true=ed_true,
        )

        # (3) per activation: real vs complex
        per_act_dir = cmp_j2_dir / "compare_real_vs_complex_per_activation"
        ensure_dir(per_act_dir)

        for act_name in act_names:
            h_r = results[("real", J2, act_name)]
            h_c = results[("complex", J2, act_name)]
            plot_three_variants_energy_curve(
                outdir=per_act_dir,
                base_name=f"real_vs_complex_act_{act_name}_J2_{J2:.2f}",
                title_base=f"REAL vs COMPLEX | act={act_name} | MLP (N,N) | L={args.L} | J1={args.J1} | J2={J2} | samples={args.n_samples}",
                iters=h_r["iters"],
                curves=[
                    ("real params", h_r["e_site"], h_r["e_site_err"]),
                    ("complex params", h_c["e_site"], h_c["e_site_err"]),
                ],
                paper_true=paper_true,
                ed_true=ed_true,
            )

            # Build overall row (side-by-side) for (act,J2)
            real_e = float(h_r["e_site"][-1]); real_err = float(h_r["e_site_err"][-1])
            cplx_e = float(h_c["e_site"][-1]); cplx_err = float(h_c["e_site_err"][-1])

            def delta(a: float, b: Optional[float]) -> Optional[float]:
                return None if b is None else a - float(b)

            overall_rows.append({
                "activation": act_name,
                "J2": J2,
                "paper_true": paper_true,
                "ed_true": ed_true,
                "real_e": real_e,
                "real_err": real_err,
                "complex_e": cplx_e,
                "complex_err": cplx_err,
                "d_real_paper": delta(real_e, paper_true),
                "d_complex_paper": delta(cplx_e, paper_true),
                "d_real_ed": delta(real_e, ed_true),
                "d_complex_ed": delta(cplx_e, ed_true),
            })

        # (4) all curves (real+complex) for this J2
        curves_all = []
        it_ref = results[("real", J2, act_names[0])]["iters"]
        for act_name in act_names:
            h_r = results[("real", J2, act_name)]
            h_c = results[("complex", J2, act_name)]
            curves_all.append((f"{act_name} (real)", h_r["e_site"], h_r["e_site_err"]))
            curves_all.append((f"{act_name} (complex)", h_c["e_site"], h_c["e_site_err"]))

        plot_three_variants_energy_curve(
            outdir=cmp_j2_dir,
            base_name=f"compare_all_real_and_complex_J2_{J2:.2f}",
            title_base=f"All activations: REAL+COMPLEX | MLP (N,N) | L={args.L} | J1={args.J1} | J2={J2} | samples={args.n_samples}",
            iters=it_ref,
            curves=curves_all,
            paper_true=paper_true,
            ed_true=ed_true,
        )

    # Write summaries
    write_per_run_summary(out_root, per_run_rows)
    write_overall_results(out_root, overall_rows)

    print("\n===================================================")
    print("DONE")
    print("===================================================")
    print(f"Outputs saved to: {out_root.resolve()}")
    print("Summary files:")
    print("  - per_run_summary.csv / per_run_summary.txt   (one row per run: dtype, act, J2)")
    print("  - overall_results.csv / overall_results.txt   (one row per (act,J2): real vs complex)")
    print("Plots:")
    print("  - per-run plots in MLP_activations_real/ and MLP_activations_complex/")
    print("  - per-J2 comparisons in compare_per_j2/")
    print("Note: NetKet ED reference is skipped automatically unless N_sites <= ed_max_sites.\n")


if __name__ == "__main__":
    main()