#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Activation-function sweep for MLP Neural Quantum States (NetKet) on J1-J2 Heisenberg model.

Separable workflow:
  - mode=train: run ONE configuration (J2, activation, dtype) and write history CSV + meta + plots
  - mode=aggregate: scan output folders, merge results, write summaries, and generate comparison plots

Fixed model structure:
  - MLP with two hidden layers (h, h), h = n_sites * hidden_scale
  - SR (VMC with SR preconditioner when available)
  - For complex runs:
      complex_activation_mode=split applies f(Re)+i f(Im) for non-holomorphic activations
      complex_activation_mode=native applies activation directly to complex (may fail for relu/silu/gelu)
"""

import os
import json
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple

import numpy as np

# Headless matplotlib support (important on clusters)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Platform selection MUST happen before importing jax/netket
# -----------------------------
def preparse_platform(argv: List[str]) -> str:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--platform", type=str, default="auto", choices=["auto", "cpu", "gpu", "tpu"])
    args, _ = p.parse_known_args(argv)
    return args.platform

def set_platform(platform: str) -> None:
    if platform.lower() != "auto":
        os.environ["JAX_PLATFORM_NAME"] = platform.lower()

_platform = preparse_platform(os.sys.argv[1:])
set_platform(_platform)

import jax
import jax.numpy as jnp

# Essential: enable 64-bit precision
jax.config.update("jax_enable_x64", True)

import netket as nk

# log_cosh import across NetKet versions
try:
    from netket.nn import log_cosh
except ImportError:
    try:
        from netket.nn.activation import log_cosh
    except ImportError:
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

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

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

def build_vmc_sr_driver(hamiltonian, vstate, optimizer, diag_shift: float):
    # Prefer modern NetKet: VMC + SR preconditioner
    if hasattr(nk.optimizer, "SR"):
        sr = nk.optimizer.SR(diag_shift=diag_shift)
        return nk.driver.VMC(
            hamiltonian=hamiltonian,
            optimizer=optimizer,
            variational_state=vstate,
            preconditioner=sr,
        )
    # Fallback older NetKet
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
def maybe_exact_ground_state_energy_per_site(L: int, J1: float, J2: float, ed_max_sites: int) -> Optional[float]:
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
                arr = np.asarray(res)
                e0 = float(arr[0] if arr.size > 0 else arr)
            return e0 / n_sites
        if hasattr(nk.exact, "diag"):
            evals = nk.exact.diag(ham)
            e0 = float(np.min(np.asarray(evals)))
            return e0 / n_sites
    except Exception as e:
        print(f"Warning: ED failed: {e}")
        return None
    return None


# -----------------------------
# Activations
# -----------------------------
def complexify_activation(act_real: Callable) -> Callable:
    def act(z):
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
    mlp_lr: float = 1e-3
    hidden_scale: int = 1
    param_dtype: Any = jnp.float64


def build_mlp_model_two_layers(n_sites: int, hidden_scale: int, param_dtype: Any, activation: Callable):
    h = int(n_sites * hidden_scale)
    return nk.models.MLP(
        hidden_dims=(h, h),
        param_dtype=param_dtype,
        hidden_activations=activation,
        output_activation=None,
        use_output_bias=True,
    )


# -----------------------------
# Plot helpers
# -----------------------------
def _plot_energy_curve(outpath: Path, title: str, iters: np.ndarray,
                      curves: List[Tuple[str, np.ndarray, np.ndarray]],
                      true_line: Optional[Tuple[str, float]] = None):
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

def plot_three_variants_energy_curve(outdir: Path, base_name: str, title_base: str,
                                    iters: np.ndarray, curves: List[Tuple[str, np.ndarray, np.ndarray]],
                                    paper_true: Optional[float], ed_true: Optional[float]):
    _plot_energy_curve(outdir / f"{base_name}.png", title_base, iters, curves, None)
    if paper_true is not None:
        _plot_energy_curve(outdir / f"{base_name}__paper_true.png",
                           f"{title_base} | Paper reference", iters, curves,
                           ("Paper E/site", float(paper_true)))
    if ed_true is not None:
        _plot_energy_curve(outdir / f"{base_name}__ed_true.png",
                           f"{title_base} | NetKet ED reference", iters, curves,
                           ("NetKet ED E/site", float(ed_true)))


# -----------------------------
# Training: run ONE configuration
# -----------------------------
def run_single_mlp_activation(cfg: RunConfig, dtype_label: str, act_name: str, act_fn: Callable, outdir: Path) -> Dict[str, np.ndarray]:
    ensure_dir(outdir)

    lattice, hilbert, ham = make_lattice_and_hamiltonian(cfg.L, cfg.J1, cfg.J2)
    n_sites = lattice.n_nodes

    sampler = build_sampler(hilbert, lattice, cfg.d_max)
    model = build_mlp_model_two_layers(n_sites=n_sites, hidden_scale=cfg.hidden_scale, param_dtype=cfg.param_dtype, activation=act_fn)
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
    if "Energy" not in data:
        raise RuntimeError("Energy not found in log data.")
    e_log = data["Energy"]

    if isinstance(e_log, dict):
        iters = np.asarray(e_log["iters"], dtype=int)
        E_mean = np.asarray(e_log["Mean"])
        E_sigma = np.asarray(e_log["Sigma"])
        if np.iscomplexobj(E_mean): E_mean = E_mean.real
        if np.iscomplexobj(E_sigma): E_sigma = E_sigma.real
        E_mean = E_mean.astype(float)
        E_sigma = E_sigma.astype(float)
    else:
        iters = np.asarray(e_log.iters, dtype=int)
        E_mean = np.asarray(e_log.Mean.real, dtype=float)
        E_sigma = np.asarray(e_log.Sigma.real, dtype=float)

    e_site = E_mean / n_sites
    e_site_err = E_sigma / n_sites

    # History CSV
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
# Aggregation: load histories from disk
# -----------------------------
def read_history_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # skip comment lines starting with '#'
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            if line.startswith("iter,"):
                continue
            rows.append(line.strip().split(","))
    arr = np.asarray(rows, dtype=float)
    iters = arr[:, 0].astype(int)
    e_site = arr[:, 3].astype(float)
    e_site_err = arr[:, 4].astype(float)
    return iters, e_site, e_site_err


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
            paper = "" if r["paper_true"] is None else f"{r['paper_true']:.6f}"
            ed = "" if r["ed_true"] is None else f"{r['ed_true']:.12f}"
            f.write(
                f"{r['dtype']},{r['activation']},{r['J2']:.2f},"
                f"{r['n_params']},{r['runtime_s']:.6f},"
                f"{r['final_e_site']:.12f},{r['final_e_site_err']:.12f},"
                f"{paper},{ed}\n"
            )

def write_overall_results(outdir: Path, rows: List[Dict[str, Any]]):
    ensure_dir(outdir)
    csv_path = outdir / "overall_results.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("# Overall results: one row per (activation,J2), real vs complex side-by-side\n")
        f.write("activation,J2,paper_true,ed_true,real_e,real_err,complex_e,complex_err,d_real_paper,d_complex_paper,d_real_ed,d_complex_ed\n")
        for r in rows:
            def s(x, prec=12):
                return "" if x is None else f"{x:.{prec}f}"
            paper = "" if r["paper_true"] is None else f"{r['paper_true']:.6f}"
            f.write(
                f"{r['activation']},{r['J2']:.2f},"
                f"{paper},"
                f"{s(r['ed_true'])},"
                f"{s(r['real_e'])},{s(r['real_err'])},"
                f"{s(r['complex_e'])},{s(r['complex_err'])},"
                f"{s(r['d_real_paper'])},{s(r['d_complex_paper'])},"
                f"{s(r['d_real_ed'])},{s(r['d_complex_ed'])}\n"
            )


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Activation sweep for MLP (two hidden layers) with real vs complex params on J1-J2 model.")

    parser.add_argument("--mode", type=str, default="train", choices=["train", "aggregate"])

    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--J1", type=float, default=1.0)

    # For train: provide single J2; for aggregate: can scan all, but you can also limit via list.
    parser.add_argument("--J2", type=float, default=0.5)
    parser.add_argument("--J2_list", type=str, default="0.5,0.6")

    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--activations", type=str, default="relu,silu,gelu,log_cosh")

    parser.add_argument("--dtype", type=str, default="real", choices=["real", "complex"])
    parser.add_argument("--complex_activation_mode", type=str, default="split", choices=["split", "native"])

    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_iter", type=int, default=600)
    parser.add_argument("--discard", type=int, default=50)
    parser.add_argument("--diag_shift", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--hidden_scale", type=int, default=1)
    parser.add_argument("--d_max", type=int, default=2)

    parser.add_argument("--platform", type=str, default=_platform, choices=["auto", "cpu", "gpu", "tpu"])
    parser.add_argument("--out", type=str, default="results_mlp_activation_sweep")

    parser.add_argument("--ed_max_sites", type=int, default=20, help="Compute NetKet ED reference only if N_sites <= ed_max_sites.")

    args = parser.parse_args()

    act_map = get_activation_map()
    out_root = Path(args.out)
    ensure_dir(out_root)

    # Save a top-level config (useful for audit/tracking)
    if args.mode == "train":
        save_json(out_root / "train_config_last.json", vars(args))
    else:
        save_json(out_root / "aggregate_config_last.json", vars(args))

    if args.mode == "train":
        act_name = args.activation.strip()
        if act_name not in act_map:
            raise ValueError(f"Unknown activation '{act_name}'. Allowed: {list(act_map.keys())}")

        dtype_label = args.dtype
        dtype = jnp.float64 if dtype_label == "real" else jnp.complex128

        base_act = act_map[act_name]
        if dtype_label == "complex":
            if args.complex_activation_mode == "split":
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
            L=args.L, J1=args.J1, J2=args.J2,
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

        n_sites = args.L * args.L
        run_dir = out_root / f"J2_{args.J2:.2f}" / f"act_{act_name}" / f"dtype_{dtype_label}"
        tag = f"MLP_{dtype_label}_{act_name}_J2_{args.J2:.2f}"

        paper_true = PAPER_TRUE_E_SITE.get(args.J2, None)
        ed_true = maybe_exact_ground_state_energy_per_site(args.L, args.J1, args.J2, args.ed_max_sites)

        print(f"[TRAIN] {tag}")
        print(f"  activation_mode: {act_note}")
        print(f"  outdir: {run_dir}")
        hist = run_single_mlp_activation(cfg, dtype_label=dtype_label, act_name=act_name, act_fn=act_fn, outdir=run_dir)

        # per-run plots
        title_base = (
            f"MLP (N,N) | act={act_name} | dtype={dtype_label} | hidden_scale={args.hidden_scale} | "
            f"L={args.L} (N_sites={n_sites}) | J1={args.J1} | J2={args.J2} | samples={args.n_samples}"
        )
        plot_three_variants_energy_curve(
            outdir=run_dir,
            base_name=f"{tag}_energy_per_site",
            title_base=title_base,
            iters=hist["iters"],
            curves=[(f"{act_name}", hist["e_site"], hist["e_site_err"])],
            paper_true=paper_true,
            ed_true=ed_true,
        )

        print(f"[DONE] Final E/site: {fmt_pm(float(hist['e_site'][-1]), float(hist['e_site_err'][-1]))}")
        return

    # ---------------------------
    # AGGREGATE MODE
    # ---------------------------
    J2_list = [float(x.strip()) for x in args.J2_list.split(",") if x.strip()]
    act_names = [x.strip() for x in args.activations.split(",") if x.strip()]
    for a in act_names:
        if a not in act_map:
            raise ValueError(f"Unknown activation '{a}'. Allowed: {list(act_map.keys())}")

    results: Dict[Tuple[str, float, str], Dict[str, np.ndarray]] = {}
    per_run_rows: List[Dict[str, Any]] = []

    for J2 in J2_list:
        paper_true = PAPER_TRUE_E_SITE.get(J2, None)
        ed_true = maybe_exact_ground_state_energy_per_site(args.L, args.J1, J2, args.ed_max_sites)

        for act in act_names:
            for dtype_label in ["real", "complex"]:
                run_dir = out_root / f"J2_{J2:.2f}" / f"act_{act}" / f"dtype_{dtype_label}"
                hist_csv = run_dir / f"mlp_{dtype_label}_{act}_J2_{J2:.2f}_history.csv"
                meta_path = run_dir / "run_meta.json"

                if not hist_csv.exists() or not meta_path.exists():
                    continue

                iters, e_site, e_site_err = read_history_csv(hist_csv)
                meta = load_json(meta_path)

                results[(dtype_label, J2, act)] = {
                    "iters": iters,
                    "e_site": e_site,
                    "e_site_err": e_site_err,
                    "n_params": np.array([int(meta.get("n_parameters", 0))], dtype=np.int64),
                    "runtime_s": np.array([float(meta.get("runtime_seconds", 0.0))], dtype=np.float64),
                }

                per_run_rows.append({
                    "dtype": dtype_label,
                    "activation": act,
                    "J2": J2,
                    "n_params": int(meta.get("n_parameters", 0)),
                    "runtime_s": float(meta.get("runtime_seconds", 0.0)),
                    "final_e_site": float(meta.get("final_energy_per_site", e_site[-1])),
                    "final_e_site_err": float(meta.get("final_energy_per_site_err", e_site_err[-1])),
                    "paper_true": paper_true,
                    "ed_true": ed_true,
                })

    # Write per-run summary
    write_per_run_summary(out_root, per_run_rows)

    # Build overall (activation,J2) rows
    overall_rows: List[Dict[str, Any]] = []
    for J2 in J2_list:
        paper_true = PAPER_TRUE_E_SITE.get(J2, None)
        ed_true = maybe_exact_ground_state_energy_per_site(args.L, args.J1, J2, args.ed_max_sites)

        for act in act_names:
            if ("real", J2, act) not in results or ("complex", J2, act) not in results:
                continue

            h_r = results[("real", J2, act)]
            h_c = results[("complex", J2, act)]

            real_e = float(h_r["e_site"][-1])
            real_err = float(h_r["e_site_err"][-1])
            cplx_e = float(h_c["e_site"][-1])
            cplx_err = float(h_c["e_site_err"][-1])

            def delta(a: float, b: Optional[float]) -> Optional[float]:
                return None if b is None else a - float(b)

            overall_rows.append({
                "activation": act,
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

    write_overall_results(out_root, overall_rows)

    # Comparison plots per J2
    cmp_root = out_root / "compare_per_j2"
    ensure_dir(cmp_root)

    for J2 in J2_list:
        paper_true = PAPER_TRUE_E_SITE.get(J2, None)
        ed_true = maybe_exact_ground_state_energy_per_site(args.L, args.J1, J2, args.ed_max_sites)

        cmp_j2_dir = cmp_root / f"J2_{J2:.2f}"
        ensure_dir(cmp_j2_dir)

        # real: compare activations
        curves_real = []
        it_ref = None
        for act in act_names:
            k = ("real", J2, act)
            if k not in results:
                continue
            h = results[k]
            if it_ref is None:
                it_ref = h["iters"]
            curves_real.append((act, h["e_site"], h["e_site_err"]))
        if curves_real and it_ref is not None:
            plot_three_variants_energy_curve(
                outdir=cmp_j2_dir,
                base_name=f"compare_activations_real_J2_{J2:.2f}",
                title_base=f"Activation comparison (REAL params) | MLP (N,N) | L={args.L} | J1={args.J1} | J2={J2}",
                iters=it_ref,
                curves=curves_real,
                paper_true=paper_true,
                ed_true=ed_true,
            )

        # complex: compare activations
        curves_c = []
        it_ref = None
        for act in act_names:
            k = ("complex", J2, act)
            if k not in results:
                continue
            h = results[k]
            if it_ref is None:
                it_ref = h["iters"]
            curves_c.append((act, h["e_site"], h["e_site_err"]))
        if curves_c and it_ref is not None:
            plot_three_variants_energy_curve(
                outdir=cmp_j2_dir,
                base_name=f"compare_activations_complex_J2_{J2:.2f}",
                title_base=f"Activation comparison (COMPLEX params) | MLP (N,N) | L={args.L} | J1={args.J1} | J2={J2}",
                iters=it_ref,
                curves=curves_c,
                paper_true=paper_true,
                ed_true=ed_true,
            )

        # per activation: real vs complex
        per_act_dir = cmp_j2_dir / "compare_real_vs_complex_per_activation"
        ensure_dir(per_act_dir)
        for act in act_names:
            kr = ("real", J2, act)
            kc = ("complex", J2, act)
            if kr not in results or kc not in results:
                continue
            h_r = results[kr]
            h_c = results[kc]
            plot_three_variants_energy_curve(
                outdir=per_act_dir,
                base_name=f"real_vs_complex_act_{act}_J2_{J2:.2f}",
                title_base=f"REAL vs COMPLEX | act={act} | MLP (N,N) | L={args.L} | J1={args.J1} | J2={J2}",
                iters=h_r["iters"],
                curves=[
                    ("real params", h_r["e_site"], h_r["e_site_err"]),
                    ("complex params", h_c["e_site"], h_c["e_site_err"]),
                ],
                paper_true=paper_true,
                ed_true=ed_true,
            )

        # all curves (real+complex) for this J2
        curves_all = []
        it_ref = None
        for act in act_names:
            kr = ("real", J2, act)
            kc = ("complex", J2, act)
            if kr in results:
                h = results[kr]
                if it_ref is None:
                    it_ref = h["iters"]
                curves_all.append((f"{act} (real)", h["e_site"], h["e_site_err"]))
            if kc in results:
                h = results[kc]
                if it_ref is None:
                    it_ref = h["iters"]
                curves_all.append((f"{act} (complex)", h["e_site"], h["e_site_err"]))
        if curves_all and it_ref is not None:
            plot_three_variants_energy_curve(
                outdir=cmp_j2_dir,
                base_name=f"compare_all_real_and_complex_J2_{J2:.2f}",
                title_base=f"All activations: REAL+COMPLEX | MLP (N,N) | L={args.L} | J1={args.J1} | J2={J2}",
                iters=it_ref,
                curves=curves_all,
                paper_true=paper_true,
                ed_true=ed_true,
            )

    print(f"[AGGREGATE DONE] Outputs written to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
