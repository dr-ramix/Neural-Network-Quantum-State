#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare RBM with REAL vs COMPLEX parameters for the J1-J2 Heisenberg model on a 6x6 square lattice.

- J1 fixed to 1
- J2 in [0.4, 0.5, 0.6]
- n_samples = 10000 (MetropolisExchange)
- Saves:
  * per-run history CSV
  * per-run energy-per-site training curve PNG
  * per-J2 comparison plot (real vs complex) PNG
  * final summary plot vs J2
  * summary.csv + summary.txt + sweep_config.json

Usage examples:
  python compare_rbm_real_vs_complex.py --platform gpu
  python compare_rbm_real_vs_complex.py --platform gpu --n_iter 800 --seed 0
"""

import os
import json
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

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
# Utilities
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def format_pm(val: float, err: float, prec: int = 6) -> str:
    return f"{val:.{prec}f} ± {err:.{prec}f}"


def style_matplotlib():
    plt.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 300,
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


def make_lattice_and_hamiltonian(L: int, J1: float, J2: float):
    lattice = nk.graph.Square(length=L, max_neighbor_order=2, pbc=True)
    hilbert = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=lattice.n_nodes)

    # J=[J1, J2] corresponds to nearest (color 0) and next-nearest (color 1) on max_neighbor_order=2 square.
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

    # Newer NetKet style:
    # SR is a preconditioner; name can vary slightly across versions.
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

    # Sampler
    d_max: int = 2

    # RBM
    rbm_alpha: int = 2
    rbm_lr: float = 1e-2


def run_rbm_vmc_sr(
    cfg: RunConfig,
    outdir: Path,
    *,
    param_dtype: Any,
    tag: str,
) -> Dict[str, np.ndarray]:
    """
    Runs one RBM optimization and writes:
      - history CSV
      - run_meta.json
      - training curve PNG

    Returns arrays for comparison plots.
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

    data = log.data
    E_hist = data["Energy"]

    iters = np.asarray(E_hist.iters)
    E_mean = np.asarray(E_hist.Mean.real)
    E_sigma = np.asarray(E_hist.Sigma.real)

    e_site = E_mean / n_sites
    e_site_err = E_sigma / n_sites

    # ---- History CSV ----
    csv_path = outdir / f"rbm_{tag.lower()}_J2_{cfg.J2:.2f}_history.csv"
    ensure_dir(csv_path.parent)
    header = "iter,energy_mean,energy_sigma,energy_per_site_mean,energy_per_site_sigma\n"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(header)
        for i, em, es, eps, epe in zip(iters, E_mean, E_sigma, e_site, e_site_err):
            f.write(f"{int(i)},{em:.12f},{es:.12f},{eps:.12f},{epe:.12f}\n")

    # ---- Meta JSON ----
    meta = {
        "tag": tag,
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
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "n_parameters": int(vstate.n_parameters),
        "runtime_seconds": float(t1 - t0),
    }
    save_json(outdir / "run_meta.json", meta)

    # ---- Training curve plot ----
    style_matplotlib()
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(iters, e_site, label=f"RBM {tag} (E/site)")
    ax.fill_between(iters, e_site - e_site_err, e_site + e_site_err, alpha=0.25)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy per site")
    ax.set_title(f"RBM {tag} | L={cfg.L} (N={n_sites}) | J1={cfg.J1} | J2={cfg.J2} | samples={cfg.n_samples}")
    ax.legend(loc="best")
    fig.tight_layout()
    png_path = outdir / f"rbm_{tag.lower()}_J2_{cfg.J2:.2f}_energy_per_site.png"
    fig.savefig(png_path)
    plt.close(fig)

    return {
        "iters": iters,
        "energy": E_mean,
        "energy_err": E_sigma,
        "e_site": e_site,
        "e_site_err": e_site_err,
        "n_sites": np.array([n_sites], dtype=np.int64),
        "runtime_s": np.array([t1 - t0], dtype=np.float64),
        "n_params": np.array([int(vstate.n_parameters)], dtype=np.int64),
    }


def plot_real_vs_complex_for_j2(
    outdir: Path,
    L: int,
    J1: float,
    J2: float,
    n_samples: int,
    hist_real: Dict[str, np.ndarray],
    hist_cplx: Dict[str, np.ndarray],
):
    ensure_dir(outdir)
    style_matplotlib()

    it_r = hist_real["iters"]
    it_c = hist_cplx["iters"]

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(it_r, hist_real["e_site"], label="RBM Real (E/site)")
    ax.fill_between(
        it_r,
        hist_real["e_site"] - hist_real["e_site_err"],
        hist_real["e_site"] + hist_real["e_site_err"],
        alpha=0.20,
    )

    ax.plot(it_c, hist_cplx["e_site"], label="RBM Complex (E/site)")
    ax.fill_between(
        it_c,
        hist_cplx["e_site"] - hist_cplx["e_site_err"],
        hist_cplx["e_site"] + hist_cplx["e_site_err"],
        alpha=0.20,
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy per site")
    ax.set_title(f"RBM Real vs Complex | L={L} | J1={J1} | J2={J2} | samples={n_samples}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / f"compare_RBM_real_vs_complex_J2_{J2:.2f}.png")
    plt.close(fig)


def plot_final_summary_vs_j2(
    outdir: Path,
    J2_list: List[float],
    real_final: np.ndarray,
    real_final_err: np.ndarray,
    cplx_final: np.ndarray,
    cplx_final_err: np.ndarray,
    L: int,
    J1: float,
    n_samples: int,
    n_iter: int,
):
    ensure_dir(outdir)
    style_matplotlib()

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.errorbar(J2_list, real_final, yerr=real_final_err, fmt="o-", label="RBM Real final E/site")
    ax.errorbar(J2_list, cplx_final, yerr=cplx_final_err, fmt="o-", label="RBM Complex final E/site")

    ax.set_xlabel("J2")
    ax.set_ylabel("Final energy per site")
    ax.set_title(f"Final Energy per Site vs J2 | RBM Real vs Complex | L={L} | J1={J1} | samples={n_samples} | iters={n_iter}")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(outdir / "final_energy_per_site_vs_J2_real_vs_complex.png")
    plt.close(fig)


def write_summary_files(outdir: Path, rows: List[Dict[str, Any]]):
    ensure_dir(outdir)

    csv_path = outdir / "summary.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("ansatz,J2,final_energy_per_site,final_energy_per_site_err,n_parameters,runtime_seconds\n")
        for r in rows:
            f.write(
                f"{r['ansatz']},{r['J2']:.2f},{r['final_e_site']:.12f},"
                f"{r['final_e_site_err']:.12f},{r['n_params']},{r['runtime_s']:.6f}\n"
            )

    headers = ["ANSATZ", "J2", "FINAL E/SITE", "ERR E/SITE", "N_PARAMS", "RUNTIME(s)"]
    table = []
    for r in rows:
        table.append([
            r["ansatz"],
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
        return "+" + "+".join([ch * (w + 2) for w in colw]) + "+\n"

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
        description="RBM Real vs Complex comparison for J1-J2 Heisenberg on square lattice (NetKet)."
    )
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--J1", type=float, default=1.0)
    parser.add_argument("--J2_list", type=str, default="0.4,0.5,0.6")

    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_iter", type=int, default=600)
    parser.add_argument("--discard", type=int, default=50)
    parser.add_argument("--diag_shift", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--rbm_alpha", type=int, default=4)
    parser.add_argument("--rbm_lr", type=float, default=1e-2)

    parser.add_argument("--platform", type=str, default=_platform,
                        choices=["auto", "cpu", "gpu", "tpu"],
                        help="JAX platform selection. Use 'auto' to let JAX decide.")

    parser.add_argument("--out", type=str, default="results_rbm_real_vs_complex")
    args = parser.parse_args()

    # Parse J2 list
    J2_list = [float(x.strip()) for x in args.J2_list.split(",") if x.strip() != ""]
    out_root = Path(args.out)
    ensure_dir(out_root)

    print("\n=== JAX Runtime ===")
    print("Requested platform:", args.platform)
    print("Backend:", jax.default_backend())
    print("Devices:", ", ".join([str(d) for d in jax.devices()]))
    print("===================\n")

    # Master sweep config
    sweep_cfg = {
        "L": args.L,
        "J1": args.J1,
        "J2_list": J2_list,
        "n_samples": args.n_samples,
        "n_iter": args.n_iter,
        "discard": args.discard,
        "diag_shift": args.diag_shift,
        "seed": args.seed,
        "rbm": {
            "alpha": args.rbm_alpha,
            "lr": args.rbm_lr,
        },
        "compare": {
            "real_param_dtype": "float64",
            "complex_param_dtype": "complex128",
        },
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
    }
    save_json(out_root / "sweep_config.json", sweep_cfg)

    # Collect final values
    rows: List[Dict[str, Any]] = []
    real_final, real_final_err = [], []
    cplx_final, cplx_final_err = [], []

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
            d_max=2,
            rbm_alpha=args.rbm_alpha,
            rbm_lr=args.rbm_lr,
        )

        print(f"=== J2 = {J2:.2f} ===")

        # REAL parameters
        real_dir = out_root / "RBM_REAL" / f"J2_{J2:.2f}"
        hist_real = run_rbm_vmc_sr(
            cfg,
            real_dir,
            param_dtype=jnp.float64,
            tag="REAL",
        )

        # COMPLEX parameters
        cplx_dir = out_root / "RBM_COMPLEX" / f"J2_{J2:.2f}"
        hist_cplx = run_rbm_vmc_sr(
            cfg,
            cplx_dir,
            param_dtype=jnp.complex128,
            tag="COMPLEX",
        )

        # Per-J2 comparison plot
        comp_dir = out_root / "compare_per_J2"
        plot_real_vs_complex_for_j2(
            outdir=comp_dir,
            L=args.L,
            J1=args.J1,
            J2=J2,
            n_samples=args.n_samples,
            hist_real=hist_real,
            hist_cplx=hist_cplx,
        )

        # Final numbers
        real_e = float(hist_real["e_site"][-1])
        real_eerr = float(hist_real["e_site_err"][-1])
        cplx_e = float(hist_cplx["e_site"][-1])
        cplx_eerr = float(hist_cplx["e_site_err"][-1])

        real_final.append(real_e)
        real_final_err.append(real_eerr)
        cplx_final.append(cplx_e)
        cplx_final_err.append(cplx_eerr)

        rows.append({
            "ansatz": "RBM_REAL",
            "J2": J2,
            "final_e_site": real_e,
            "final_e_site_err": real_eerr,
            "n_params": int(hist_real["n_params"][0]),
            "runtime_s": float(hist_real["runtime_s"][0]),
        })
        rows.append({
            "ansatz": "RBM_COMPLEX",
            "J2": J2,
            "final_e_site": cplx_e,
            "final_e_site_err": cplx_eerr,
            "n_params": int(hist_cplx["n_params"][0]),
            "runtime_s": float(hist_cplx["runtime_s"][0]),
        })

        print("Final (energy per site):")
        print(f"  RBM Real   : {format_pm(real_e, real_eerr, prec=6)}")
        print(f"  RBM Complex: {format_pm(cplx_e, cplx_eerr, prec=6)}\n")

    # Write summary csv/txt + final plot
    write_summary_files(out_root, rows)

    plot_final_summary_vs_j2(
        outdir=out_root,
        J2_list=J2_list,
        real_final=np.array(real_final),
        real_final_err=np.array(real_final_err),
        cplx_final=np.array(cplx_final),
        cplx_final_err=np.array(cplx_final_err),
        L=args.L,
        J1=args.J1,
        n_samples=args.n_samples,
        n_iter=args.n_iter,
    )

    print("\n=== DONE ===")
    print(f"All outputs saved to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
