#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import netket as nk
import netket.nn as nknn


# -----------------------------
# Utilities
# -----------------------------
def set_platform(platform: str) -> None:
    """
    platform: 'auto' | 'cpu' | 'gpu' | 'tpu'
    If auto: do not override JAX platform selection.
    """
    if platform.lower() != "auto":
        os.environ["JAX_PLATFORM_NAME"] = platform.lower()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_float(x) -> float:
    return float(np.asarray(x).item())


def format_pm(val: float, err: float, width: int = 0, prec: int = 6) -> str:
    s = f"{val:.{prec}f} ± {err:.{prec}f}"
    return s.rjust(width) if width > 0 else s


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def style_matplotlib():
    # Clean, readable defaults (no external style deps)
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 2.0,
    })


@dataclass
class RunConfig:
    L: int = 6
    J1: float = 1.0
    J2: float = 0.5
    n_samples: int = 10_000
    n_discard_per_chain: int = 50
    n_iter: int = 600
    diag_shift: float = 0.01
    seed: int = 1234

    # Sampler
    d_max: int = 2

    # MLP model
    mlp_hidden_scale: int = 1  # hidden dims = (n_sites*scale, n_sites*scale)

    # RBM model
    rbm_alpha: int = 4

    # Optimizers (your choices kept, but you can edit)
    mlp_lr: float = 1e-3
    rbm_lr: float = 1e-2


def make_lattice_and_hamiltonian(L: int, J1: float, J2: float):
    lattice = nk.graph.Square(length=L, max_neighbor_order=2, pbc=True)
    hilbert = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=lattice.n_nodes)

    # Heisenberg with 1st & 2nd neighbor couplings (NetKet uses edge colors)
    hamiltonian = nk.operator.Heisenberg(
        hilbert=hilbert,
        graph=lattice,
        J=[J1, J2],
        sign_rule=[False, False],
    )
    return lattice, hilbert, hamiltonian


def build_mlp_model(n_sites: int, hidden_scale: int = 1):
    h = n_sites * hidden_scale
    model = nk.models.MLP(
        hidden_dims=(h, h),
        param_dtype=jnp.complex128,
        hidden_activations=nknn.log_cosh,
        output_activation=None,
        use_output_bias=True,
    )
    return model


def build_rbm_model(alpha: int = 4):
    return nk.models.RBM(alpha=alpha, use_hidden_bias=True, use_visible_bias=True)


def build_sampler(hilbert, lattice, d_max: int = 2):
    return nk.sampler.MetropolisExchange(hilbert=hilbert, graph=lattice, d_max=d_max)


def run_single_vmc_sr(
    arch: str,
    cfg: RunConfig,
    outdir: Path,
) -> Dict[str, np.ndarray]:
    """
    Runs one optimization for a given architecture and J2.
    Returns a dict with iteration history arrays.
    Saves:
      - runtime log raw json (NetKet history exported)
      - per-iteration CSV
      - plot for that run (energy per site)
    """
    ensure_dir(outdir)

    lattice, hilbert, ham = make_lattice_and_hamiltonian(cfg.L, cfg.J1, cfg.J2)
    n_sites = lattice.n_nodes

    sampler = build_sampler(hilbert, lattice, cfg.d_max)

    if arch.lower() == "mlp":
        model = build_mlp_model(n_sites, cfg.mlp_hidden_scale)
        opt = nk.optimizer.Adam(learning_rate=cfg.mlp_lr)
        tag = "MLP"
    elif arch.lower() == "rbm":
        model = build_rbm_model(cfg.rbm_alpha)
        opt = nk.optimizer.Adam(learning_rate=cfg.rbm_lr)
        tag = "RBM"
    else:
        raise ValueError(f"Unknown arch: {arch}")

    # Seed for reproducibility
    nk.jax.random.seed(cfg.seed)

    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=cfg.n_samples,
        n_discard_per_chain=cfg.n_discard_per_chain,
    )

    driver = nk.driver.VMC_SR(
        hamiltonian=ham,
        optimizer=opt,
        diag_shift=cfg.diag_shift,
        variational_state=vstate,
    )

    log = nk.logging.RuntimeLog()

    t0 = time.time()
    driver.run(n_iter=cfg.n_iter, out=log)
    t1 = time.time()

    data = log.data
    E_hist = data["Energy"]

    iters = np.asarray(E_hist.iters)
    E_mean = np.asarray(E_hist.Mean.real)
    E_sigma = np.asarray(E_hist.Sigma.real)

    # Energy per site
    e_site = E_mean / n_sites
    e_site_err = E_sigma / n_sites

    # Save per-iteration CSV
    csv_path = outdir / f"{arch.lower()}_J2_{cfg.J2:.2f}_history.csv"
    header = "iter,energy_mean,energy_sigma,energy_per_site_mean,energy_per_site_sigma\n"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(header)
        for i, em, es, eps, epe in zip(iters, E_mean, E_sigma, e_site, e_site_err):
            f.write(f"{int(i)},{em:.12f},{es:.12f},{eps:.12f},{epe:.12f}\n")

    # Save a compact run metadata json
    meta = {
        "arch": tag,
        "L": cfg.L,
        "n_sites": int(n_sites),
        "J1": cfg.J1,
        "J2": cfg.J2,
        "n_samples": cfg.n_samples,
        "n_discard_per_chain": cfg.n_discard_per_chain,
        "n_iter": cfg.n_iter,
        "diag_shift": cfg.diag_shift,
        "seed": cfg.seed,
        "optimizer": {"type": "Adam", "learning_rate": cfg.mlp_lr if arch=="mlp" else cfg.rbm_lr},
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "n_parameters": int(vstate.n_parameters),
        "runtime_seconds": t1 - t0,
    }
    save_json(outdir / "run_meta.json", meta)

    # Plot: energy per site vs iteration (with error band)
    style_matplotlib()
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(iters, e_site, label=f"{tag} (E/site)")
    ax.fill_between(iters, e_site - e_site_err, e_site + e_site_err, alpha=0.25)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy per site")
    ax.set_title(f"{tag} | L={cfg.L} (N={n_sites}) | J1={cfg.J1} | J2={cfg.J2} | samples={cfg.n_samples}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / f"{arch.lower()}_J2_{cfg.J2:.2f}_energy_per_site.png")
    plt.close(fig)

    return {
        "iters": iters,
        "energy": E_mean,
        "energy_err": E_sigma,
        "e_site": e_site,
        "e_site_err": e_site_err,
        "n_sites": np.array([n_sites]),
        "runtime_s": np.array([t1 - t0]),
        "n_params": np.array([int(vstate.n_parameters)]),
    }


def plot_mlp_vs_rbm_for_j2(
    J2: float,
    L: int,
    J1: float,
    outdir: Path,
    mlp_hist: Dict[str, np.ndarray],
    rbm_hist: Dict[str, np.ndarray],
    n_samples: int,
):
    ensure_dir(outdir)
    style_matplotlib()

    it = mlp_hist["iters"]
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(it, mlp_hist["e_site"], label="MLP (E/site)")
    ax.fill_between(it, mlp_hist["e_site"] - mlp_hist["e_site_err"], mlp_hist["e_site"] + mlp_hist["e_site_err"], alpha=0.20)

    ax.plot(it, rbm_hist["e_site"], label="RBM (E/site)")
    ax.fill_between(it, rbm_hist["e_site"] - rbm_hist["e_site_err"], rbm_hist["e_site"] + rbm_hist["e_site_err"], alpha=0.20)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy per site")
    ax.set_title(f"MLP vs RBM | L={L} | J1={J1} | J2={J2} | samples={n_samples}")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(outdir / f"compare_MLP_vs_RBM_J2_{J2:.2f}.png")
    plt.close(fig)


def plot_final_summary(
    outdir: Path,
    J2_list: List[float],
    mlp_final: np.ndarray,
    mlp_final_err: np.ndarray,
    rbm_final: np.ndarray,
    rbm_final_err: np.ndarray,
    L: int,
    J1: float,
    n_samples: int,
    n_iter: int,
):
    ensure_dir(outdir)
    style_matplotlib()

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.errorbar(J2_list, mlp_final, yerr=mlp_final_err, fmt="o-", label="MLP final E/site")
    ax.errorbar(J2_list, rbm_final, yerr=rbm_final_err, fmt="o-", label="RBM final E/site")

    ax.set_xlabel("J2")
    ax.set_ylabel("Final energy per site")
    ax.set_title(f"Final Energy per Site vs J2 | L={L} | J1={J1} | samples={n_samples} | iters={n_iter}")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(outdir / "final_energy_per_site_vs_J2.png")
    plt.close(fig)


def write_summary_files(
    outdir: Path,
    rows: List[Dict],
):
    """
    rows: list of dicts with keys:
      arch, J2, final_e_site, final_e_site_err, n_params, runtime_s
    Writes summary.csv and summary.txt
    """
    ensure_dir(outdir)

    # CSV
    csv_path = outdir / "summary.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("arch,J2,final_energy_per_site,final_energy_per_site_err,n_parameters,runtime_seconds\n")
        for r in rows:
            f.write(f"{r['arch']},{r['J2']:.2f},{r['final_e_site']:.12f},{r['final_e_site_err']:.12f},{r['n_params']},{r['runtime_s']:.6f}\n")

    # Pretty TXT table (no external deps)
    # Compute column widths
    headers = ["ARCH", "J2", "FINAL E/SITE", "ERR E/SITE", "N_PARAMS", "RUNTIME(s)"]
    table = []
    for r in rows:
        table.append([
            r["arch"],
            f"{r['J2']:.2f}",
            f"{r['final_e_site']:.8f}",
            f"{r['final_e_site_err']:.8f}",
            str(r["n_params"]),
            f"{r['runtime_s']:.2f}",
        ])

    colw = [len(h) for h in headers]
    for row in table:
        for i, cell in enumerate(row):
            colw[i] = max(colw[i], len(cell))

    def line(ch="-"):
        return "+" + "+".join([ch*(w+2) for w in colw]) + "+\n"

    txt_path = outdir / "summary.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(line("-"))
        f.write("| " + " | ".join([headers[i].ljust(colw[i]) for i in range(len(headers))]) + " |\n")
        f.write(line("="))
        for row in table:
            f.write("| " + " | ".join([row[i].ljust(colw[i]) for i in range(len(headers))]) + " |\n")
        f.write(line("-"))


def main():
    parser = argparse.ArgumentParser(
        description="Sweep J2 for J1-J2 Heisenberg on square lattice using NetKet NQS (MLP vs RBM)."
    )
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--J1", type=float, default=1.0)
    parser.add_argument("--J2_list", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--n_samples", type=int, default=10_000)
    parser.add_argument("--n_iter", type=int, default=600)
    parser.add_argument("--discard", type=int, default=50)
    parser.add_argument("--diag_shift", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--mlp_hidden_scale", type=int, default=1)

    parser.add_argument("--rbm_lr", type=float, default=1e-2)
    parser.add_argument("--rbm_alpha", type=int, default=4)

    parser.add_argument("--platform", type=str, default="auto", choices=["auto", "cpu", "gpu", "tpu"],
                        help="JAX platform selection. Use 'auto' to let JAX decide (GPU on cluster, CPU on Mac).")
    parser.add_argument("--out", type=str, default="results_j1j2_sweep")
    args = parser.parse_args()

    set_platform(args.platform)

    # Print device info nicely
    print("\n=== JAX Runtime ===")
    print("Backend:", jax.default_backend())
    print("Devices:", ", ".join([str(d) for d in jax.devices()]))
    print("===================\n")

    J2_list = [float(x.strip()) for x in args.J2_list.split(",") if x.strip() != ""]
    out_root = Path(args.out)
    ensure_dir(out_root)

    # Master metadata
    master_cfg = {
        "L": args.L,
        "J1": args.J1,
        "J2_list": J2_list,
        "n_samples": args.n_samples,
        "n_iter": args.n_iter,
        "discard": args.discard,
        "diag_shift": args.diag_shift,
        "seed": args.seed,
        "mlp": {"lr": args.mlp_lr, "hidden_scale": args.mlp_hidden_scale, "activation": "log_cosh"},
        "rbm": {"lr": args.rbm_lr, "alpha": args.rbm_alpha},
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
    }
    save_json(out_root / "sweep_config.json", master_cfg)

    summary_rows = []

    # Store finals for global plot
    mlp_final = []
    mlp_final_err = []
    rbm_final = []
    rbm_final_err = []

    for J2 in J2_list:
        cfg = RunConfig(
            L=args.L,
            J1=args.J1,
            J2=J2,
            n_samples=args.n_samples,
            n_discard_per_chain=args.discard,
            n_iter=args.n_iter,
            diag_shift=args.diag_shift,
            seed=args.seed,
            mlp_hidden_scale=args.mlp_hidden_scale,
            rbm_alpha=args.rbm_alpha,
            mlp_lr=args.mlp_lr,
            rbm_lr=args.rbm_lr,
        )

        print(f"=== J2 = {J2:.2f} ===")

        # Run MLP
        mlp_dir = out_root / "MLP" / f"J2_{J2:.2f}"
        mlp_hist = run_single_vmc_sr("mlp", cfg, mlp_dir)

        # Run RBM
        rbm_dir = out_root / "RBM" / f"J2_{J2:.2f}"
        rbm_hist = run_single_vmc_sr("rbm", cfg, rbm_dir)

        # Per-J2 comparison plot
        comp_dir = out_root / "compare_per_J2"
        plot_mlp_vs_rbm_for_j2(
            J2=J2,
            L=args.L,
            J1=args.J1,
            outdir=comp_dir,
            mlp_hist=mlp_hist,
            rbm_hist=rbm_hist,
            n_samples=args.n_samples,
        )

        # Final values
        mlp_e = float(mlp_hist["e_site"][-1])
        mlp_eerr = float(mlp_hist["e_site_err"][-1])
        rbm_e = float(rbm_hist["e_site"][-1])
        rbm_eerr = float(rbm_hist["e_site_err"][-1])

        mlp_final.append(mlp_e)
        mlp_final_err.append(mlp_eerr)
        rbm_final.append(rbm_e)
        rbm_final_err.append(rbm_eerr)

        summary_rows.append({
            "arch": "MLP",
            "J2": J2,
            "final_e_site": mlp_e,
            "final_e_site_err": mlp_eerr,
            "n_params": int(mlp_hist["n_params"][0]),
            "runtime_s": float(mlp_hist["runtime_s"][0]),
        })
        summary_rows.append({
            "arch": "RBM",
            "J2": J2,
            "final_e_site": rbm_e,
            "final_e_site_err": rbm_eerr,
            "n_params": int(rbm_hist["n_params"][0]),
            "runtime_s": float(rbm_hist["runtime_s"][0]),
        })

        # Nice console print for this J2
        print("Final (energy per site):")
        print(f"  MLP: {format_pm(mlp_e, mlp_eerr, prec=6)}")
        print(f"  RBM: {format_pm(rbm_e, rbm_eerr, prec=6)}")
        print("")

    # Write summary CSV + TXT
    write_summary_files(out_root, summary_rows)

    # Global final plot
    plot_final_summary(
        outdir=out_root,
        J2_list=J2_list,
        mlp_final=np.array(mlp_final),
        mlp_final_err=np.array(mlp_final_err),
        rbm_final=np.array(rbm_final),
        rbm_final_err=np.array(rbm_final_err),
        L=args.L,
        J1=args.J1,
        n_samples=args.n_samples,
        n_iter=args.n_iter,
    )

    print("\n=== DONE ===")
    print(f"All outputs saved to: {out_root.resolve()}")
    print("Key files:")
    print("  - summary.csv")
    print("  - summary.txt")
    print("  - final_energy_per_site_vs_J2.png")
    print("  - compare_per_J2/compare_MLP_vs_RBM_J2_*.png")
    print("  - MLP/J2_xx/ and RBM/J2_xx/ per-iteration CSV + plots")


if __name__ == "__main__":
    main()
