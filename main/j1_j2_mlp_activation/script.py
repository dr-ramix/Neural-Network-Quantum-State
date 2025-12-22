#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare MLP (N_N) with COMPLEX parameters using different activation functions
for the J1-J2 Heisenberg model on a 6x6 square lattice.

- J1 fixed to 1
- J2 in [0.5, 0.6]
- Activations: ReLU, SiLU, GELU, log_cosh
- n_samples = 10000, same VMC+SR setup
- Saves per-run:
  * history CSV
  * training curve PNG
  * run_meta.json
- Per J2:
  * comparison plot: activation functions on same figure
- Global:
  * summary.csv + summary.txt + sweep_config.json
  * final summary plot vs J2 (final E/site for each activation)

Usage:
  python compare_mlp_activations.py --platform gpu
  python compare_mlp_activations.py --platform gpu --n_iter 800 --seed 0
"""

import os
import json
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Callable, Tuple

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

# NetKet activation functions
# log_cosh is in netket.nn.activation; ReLU/SiLU/GELU are easiest via jax.nn
from netket.nn.activation import log_cosh


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


def build_mlp_model(n_sites: int, hidden_scale: int, param_dtype: Any, activation: Callable):
    h = n_sites * hidden_scale
    return nk.models.MLP(
        hidden_dims=(h, h),
        param_dtype=param_dtype,
        hidden_activations=activation,
        output_activation=None,
        use_output_bias=True,
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

    # MLP (N_N)
    mlp_hidden_scale: int = 1  # hidden dims = (n_sites*scale, n_sites*scale)
    mlp_lr: float = 1e-3

    # Force complex parameters
    param_dtype: Any = jnp.complex128


def get_activation_map() -> Dict[str, Callable]:
    """
    Activation functions requested:
      - ReLU, SiLU, GELU, log_cosh
    We'll use jax.nn for the first three and NetKet's log_cosh.
    """
    return {
        "relu": jax.nn.relu,
        "silu": jax.nn.silu,
        "gelu": jax.nn.gelu,
        "log_cosh": log_cosh,
    }


def run_single_mlp_activation(
    cfg: RunConfig,
    outdir: Path,
    act_name: str,
    act_fn: Callable,
) -> Dict[str, np.ndarray]:
    ensure_dir(outdir)

    lattice, hilbert, ham = make_lattice_and_hamiltonian(cfg.L, cfg.J1, cfg.J2)
    n_sites = lattice.n_nodes

    sampler = build_sampler(hilbert, lattice, cfg.d_max)
    model = build_mlp_model(
        n_sites=n_sites,
        hidden_scale=cfg.mlp_hidden_scale,
        param_dtype=cfg.param_dtype,
        activation=act_fn,
    )
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

    data = log.data
    E_hist = data["Energy"]

    iters = np.asarray(E_hist.iters)
    E_mean = np.asarray(E_hist.Mean.real)
    E_sigma = np.asarray(E_hist.Sigma.real)

    e_site = E_mean / n_sites
    e_site_err = E_sigma / n_sites

    # ---- History CSV ----
    csv_path = outdir / f"mlp_{act_name}_J2_{cfg.J2:.2f}_history.csv"
    ensure_dir(csv_path.parent)
    header = "iter,energy_mean,energy_sigma,energy_per_site_mean,energy_per_site_sigma\n"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(header)
        for i, em, es, eps, epe in zip(iters, E_mean, E_sigma, e_site, e_site_err):
            f.write(f"{int(i)},{em:.12f},{es:.12f},{eps:.12f},{epe:.12f}\n")

    # ---- Meta JSON ----
    meta = {
        "arch": "MLP",
        "activation": act_name,
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
        "mlp_hidden_scale": cfg.mlp_hidden_scale,
        "n_parameters": int(vstate.n_parameters),
        "runtime_seconds": float(t1 - t0),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
    }
    save_json(outdir / "run_meta.json", meta)

    # ---- Training curve PNG ----
    style_matplotlib()
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(iters, e_site, label=f"MLP ({act_name}) E/site")
    ax.fill_between(iters, e_site - e_site_err, e_site + e_site_err, alpha=0.25)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy per site")
    ax.set_title(
        f"MLP ({act_name}) | L={cfg.L} (N={n_sites}) | J1={cfg.J1} | J2={cfg.J2} | samples={cfg.n_samples}"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / f"mlp_{act_name}_J2_{cfg.J2:.2f}_energy_per_site.png")
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


def plot_compare_activations_for_j2(
    outdir: Path,
    L: int,
    J1: float,
    J2: float,
    n_samples: int,
    hists: Dict[str, Dict[str, np.ndarray]],
):
    """
    One plot per J2, comparing training curves (E/site) across activations.
    """
    ensure_dir(outdir)
    style_matplotlib()

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    # Plot each activation
    for act_name, hist in hists.items():
        it = hist["iters"]
        ax.plot(it, hist["e_site"], label=f"{act_name}")
        ax.fill_between(
            it,
            hist["e_site"] - hist["e_site_err"],
            hist["e_site"] + hist["e_site_err"],
            alpha=0.15,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy per site")
    ax.set_title(f"MLP complex | Activations comparison | L={L} | J1={J1} | J2={J2} | samples={n_samples}")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(outdir / f"compare_MLP_activations_J2_{J2:.2f}.png")
    plt.close(fig)


def plot_final_summary_vs_j2(
    outdir: Path,
    J2_list: List[float],
    finals: Dict[str, Tuple[np.ndarray, np.ndarray]],
    L: int,
    J1: float,
    n_samples: int,
    n_iter: int,
):
    """
    Final E/site vs J2, one curve per activation.
    finals[act] = (final_e_site_array, final_e_site_err_array)
    """
    ensure_dir(outdir)
    style_matplotlib()

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    for act_name, (vals, errs) in finals.items():
        ax.errorbar(J2_list, vals, yerr=errs, fmt="o-", label=f"{act_name}")

    ax.set_xlabel("J2")
    ax.set_ylabel("Final energy per site")
    ax.set_title(f"Final E/site vs J2 | MLP complex | L={L} | J1={J1} | samples={n_samples} | iters={n_iter}")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(outdir / "final_energy_per_site_vs_J2_by_activation.png")
    plt.close(fig)


def write_summary_files(outdir: Path, rows: List[Dict[str, Any]]):
    ensure_dir(outdir)

    csv_path = outdir / "summary.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("arch,activation,J2,final_energy_per_site,final_energy_per_site_err,n_parameters,runtime_seconds\n")
        for r in rows:
            f.write(
                f"{r['arch']},{r['activation']},{r['J2']:.2f},"
                f"{r['final_e_site']:.12f},{r['final_e_site_err']:.12f},"
                f"{r['n_params']},{r['runtime_s']:.6f}\n"
            )

    headers = ["ARCH", "ACTIVATION", "J2", "FINAL E/SITE", "ERR E/SITE", "N_PARAMS", "RUNTIME(s)"]
    table = []
    for r in rows:
        table.append([
            r["arch"],
            r["activation"],
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
        description="Compare MLP complex with different activation functions (NetKet VMC+SR) for J1-J2 model."
    )
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--J1", type=float, default=1.0)
    parser.add_argument("--J2_list", type=str, default="0.5,0.6")

    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_iter", type=int, default=600)
    parser.add_argument("--discard", type=int, default=50)
    parser.add_argument("--diag_shift", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--mlp_hidden_scale", type=int, default=1)

    parser.add_argument("--platform", type=str, default=_platform,
                        choices=["auto", "cpu", "gpu", "tpu"],
                        help="JAX platform selection. Use 'auto' to let JAX decide.")
    parser.add_argument("--out", type=str, default="results_mlp_activation_sweep")
    args = parser.parse_args()

    print("\n=== JAX Runtime ===")
    print("Requested platform:", args.platform)
    print("Backend:", jax.default_backend())
    print("Devices:", ", ".join([str(d) for d in jax.devices()]))
    print("===================\n")

    J2_list = [float(x.strip()) for x in args.J2_list.split(",") if x.strip() != ""]
    out_root = Path(args.out)
    ensure_dir(out_root)

    act_map = get_activation_map()
    act_names = ["relu", "silu", "gelu", "log_cosh"]  # fixed order

    sweep_cfg = {
        "L": args.L,
        "J1": args.J1,
        "J2_list": J2_list,
        "n_samples": args.n_samples,
        "n_iter": args.n_iter,
        "discard": args.discard,
        "diag_shift": args.diag_shift,
        "seed": args.seed,
        "param_dtype": "complex128",
        "mlp": {
            "lr": args.mlp_lr,
            "hidden_scale": args.mlp_hidden_scale,
            "activations": act_names,
        },
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
    }
    save_json(out_root / "sweep_config.json", sweep_cfg)

    summary_rows: List[Dict[str, Any]] = []

    # Store final E/site arrays per activation across J2
    finals: Dict[str, Tuple[List[float], List[float]]] = {a: ([], []) for a in act_names}

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
            mlp_hidden_scale=args.mlp_hidden_scale,
            mlp_lr=args.mlp_lr,
            param_dtype=jnp.complex128,
        )

        print(f"=== J2 = {J2:.2f} ===")

        # Run all activations for this J2
        per_j2_hists: Dict[str, Dict[str, np.ndarray]] = {}

        for act_name in act_names:
            act_fn = act_map[act_name]
            run_dir = out_root / "MLP" / f"J2_{J2:.2f}" / f"act_{act_name}"
            hist = run_single_mlp_activation(cfg, run_dir, act_name=act_name, act_fn=act_fn)
            per_j2_hists[act_name] = hist

            final_e = float(hist["e_site"][-1])
            final_eerr = float(hist["e_site_err"][-1])

            finals[act_name][0].append(final_e)
            finals[act_name][1].append(final_eerr)

            summary_rows.append({
                "arch": "MLP",
                "activation": act_name,
                "J2": J2,
                "final_e_site": final_e,
                "final_e_site_err": final_eerr,
                "n_params": int(hist["n_params"][0]),
                "runtime_s": float(hist["runtime_s"][0]),
            })

            print(f"  {act_name:8s} -> final E/site: {format_pm(final_e, final_eerr, prec=6)}")

        # Per-J2 comparison plot (all activations)
        comp_dir = out_root / "compare_per_J2"
        plot_compare_activations_for_j2(
            outdir=comp_dir,
            L=args.L,
            J1=args.J1,
            J2=J2,
            n_samples=args.n_samples,
            hists=per_j2_hists,
        )
        print("")

    # Write summary files
    write_summary_files(out_root, summary_rows)

    # Final summary plot vs J2
    finals_np: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for act_name in act_names:
        vals = np.array(finals[act_name][0], dtype=np.float64)
        errs = np.array(finals[act_name][1], dtype=np.float64)
        finals_np[act_name] = (vals, errs)

    plot_final_summary_vs_j2(
        outdir=out_root,
        J2_list=J2_list,
        finals=finals_np,
        L=args.L,
        J1=args.J1,
        n_samples=args.n_samples,
        n_iter=args.n_iter,
    )

    print("\n=== DONE ===")
    print(f"All outputs saved to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
