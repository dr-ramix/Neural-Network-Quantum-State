#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
J1-J2 Heisenberg (square lattice, PBC) sweep over J2 with:
  - MLP (real + complex params)
  - RBM (real + complex params)
and comparisons + high-res plots + CSV/TXT summaries.

This version adds *separable execution* + *merge/aggregate* without changing the
numerical logic:

Modes:
  --mode sweep      (default) original behavior: run everything in one process
  --mode train      run ONE configuration only: (J2, arch, dtype) -> writes history + meta + per-model plots
  --mode aggregate  do NOT train; read existing history/meta from disk and generate:
                    - per-model plots (3 variants)
                    - compare_per_j2 plots (3 variants) when data available
                    - summary.csv/.txt and overall_results.csv/.txt (missing entries left blank)

Important: keeps complex128 and float64 exactly as requested.
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
# Paper "true" values (energy per site) provided by you
# -----------------------------
PAPER_TRUE_E_SITE = {
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

def load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

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


def parse_j2_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def history_csv_path(outdir: Path, tag: str, J2: float) -> Path:
    return outdir / f"{tag}_J2_{J2:.2f}_history.csv"


def load_history_csv(csv_path: Path) -> Dict[str, np.ndarray]:
    """
    Loads the history CSV produced by run_single_vmc_sr().

    CSV format:
      # comment lines...
      iter,energy_mean,energy_sigma,energy_per_site_mean,energy_per_site_sigma
      ...
    """
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))

    # Read numeric rows (skip leading # comments)
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            if line.lower().startswith("iter,"):
                continue
            parts = line.strip().split(",")
            if len(parts) != 5:
                continue
            rows.append(parts)

    if not rows:
        raise ValueError(f"No data rows found in {csv_path}")

    arr = np.asarray(rows, dtype=float)
    iters = arr[:, 0].astype(int)
    energy_mean = arr[:, 1]
    energy_sigma = arr[:, 2]
    e_site = arr[:, 3]
    e_site_err = arr[:, 4]

    return {
        "iters": iters,
        "energy": energy_mean,
        "energy_err": energy_sigma,
        "e_site": e_site,
        "e_site_err": e_site_err,
    }


def infer_n_sites_from_L(L: int) -> int:
    return int(L * L)


# -----------------------------
# Core VMC/SR components (unchanged logic)
# -----------------------------
@dataclass
class RunConfig:
    L: int = 6
    J1: float = 1.0
    J2: float = 0.5

    n_samples: int = 10000
    n_discard_per_chain: int = 50
    n_iter: int = 800
    diag_shift: float = 0.01
    seed: int = 1234

    # Sampler
    d_max: int = 2

    # MLP
    mlp_hidden_scale: int = 1

    # RBM
    rbm_alpha: int = 4

    # Optimizers
    mlp_lr: float = 1e-3
    rbm_lr: float = 1e-3

    # dtype
    param_dtype: Any = jnp.complex128


def make_lattice_and_hamiltonian(L: int, J1: float, J2: float):
    lattice = nk.graph.Square(length=L, max_neighbor_order=2, pbc=True)
    hilbert = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=lattice.n_nodes)

    hamiltonian = nk.operator.Heisenberg(
        hilbert=hilbert,
        graph=lattice,
        J=[J1, J2],
        sign_rule=[False, False],
    )
    return lattice, hilbert, hamiltonian


def build_sampler(hilbert, lattice, d_max: int = 2):
    return nk.sampler.MetropolisExchange(hilbert=hilbert, graph=lattice, d_max=d_max)


def build_mlp_model(n_sites: int, hidden_scale: int, param_dtype):
    h = n_sites * hidden_scale
    return nk.models.MLP(
        hidden_dims=(h, h),
        param_dtype=param_dtype,
        hidden_activations=log_cosh,
        output_activation=None,
        use_output_bias=True,
    )


def build_rbm_model(alpha: int, param_dtype):
    return nk.models.RBM(
        alpha=alpha,
        use_hidden_bias=True,
        use_visible_bias=True,
        param_dtype=param_dtype,
    )


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


def run_single_vmc_sr(
    arch: str,
    cfg: RunConfig,
    outdir: Path,
    tag: str,
    lr: float,
) -> Dict[str, np.ndarray]:
    ensure_dir(outdir)

    lattice, hilbert, ham = make_lattice_and_hamiltonian(cfg.L, cfg.J1, cfg.J2)
    n_sites = lattice.n_nodes

    sampler = build_sampler(hilbert, lattice, cfg.d_max)

    arch_l = arch.lower()
    if arch_l == "mlp":
        model = build_mlp_model(n_sites, cfg.mlp_hidden_scale, cfg.param_dtype)
    elif arch_l == "rbm":
        model = build_rbm_model(cfg.rbm_alpha, cfg.param_dtype)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    opt = nk.optimizer.Adam(learning_rate=lr)

    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=cfg.n_samples,
        n_discard_per_chain=cfg.n_discard_per_chain,
        seed=cfg.seed,
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

    iters = np.asarray(E_hist.iters, dtype=int)
    E_mean = np.asarray(E_hist.Mean.real, dtype=float)
    E_sigma = np.asarray(E_hist.Sigma.real, dtype=float)

    e_site = E_mean / n_sites
    e_site_err = E_sigma / n_sites

    # History CSV
    csv_path = outdir / f"{tag}_J2_{cfg.J2:.2f}_history.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(f"# {tag} history | L={cfg.L} (N={n_sites}) | J1={cfg.J1} | J2={cfg.J2}\n")
        f.write(f"# n_samples={cfg.n_samples} | n_iter={cfg.n_iter} | diag_shift={cfg.diag_shift} | seed={cfg.seed}\n")
        f.write("iter,energy_mean,energy_sigma,energy_per_site_mean,energy_per_site_sigma\n")
        for i, em, es, eps, epe in zip(iters, E_mean, E_sigma, e_site, e_site_err):
            f.write(f"{int(i)},{em:.12f},{es:.12f},{eps:.12f},{epe:.12f}\n")

    meta = {
        "tag": tag,
        "arch": arch.upper(),
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
        "optimizer": {"type": "Adam", "learning_rate": lr},
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
        "energy": E_mean,
        "energy_err": E_sigma,
        "e_site": e_site,
        "e_site_err": e_site_err,
        "n_sites": np.array([n_sites]),
        "runtime_s": np.array([t1 - t0]),
        "n_params": np.array([int(vstate.n_parameters)]),
    }


# -----------------------------
# Plotting (unchanged)
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
# Summaries (unchanged)
# -----------------------------
def write_summary_files(outdir: Path, rows: List[Dict[str, Any]]):
    ensure_dir(outdir)

    csv_path = outdir / "summary.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("# Summary of VMC_SR runs (final energies per site)\n")
        f.write("# One row per run: (arch, dtype, J2)\n")
        f.write("arch,dtype,J2,final_energy_per_site,final_energy_per_site_err,n_parameters,runtime_seconds,paper_true_e_site,ed_true_e_site\n")
        for r in rows:
            f.write(
                f"{r['arch']},{r['dtype']},{r['J2']:.2f},"
                f"{r['final_e_site']:.12f},{r['final_e_site_err']:.12f},"
                f"{r['n_params']},{r['runtime_s']:.6f},"
                f"{'' if r['paper_true_e_site'] is None else f'{r['paper_true_e_site']:.6f}'},"
                f"{'' if r['ed_true_e_site'] is None else f'{r['ed_true_e_site']:.12f}'}\n"
            )

    headers = ["ARCH", "DTYPE", "J2", "FINAL E/SITE", "ERR", "N_PARAMS", "RUNTIME(s)", "PAPER TRUE", "ED TRUE"]
    table = []
    for r in rows:
        table.append([
            r["arch"],
            r["dtype"],
            f"{r['J2']:.2f}",
            f"{r['final_e_site']:.8f}",
            f"{r['final_e_site_err']:.8f}",
            str(r["n_params"]),
            f"{r['runtime_s']:.2f}",
            ("" if r["paper_true_e_site"] is None else f"{r['paper_true_e_site']:.5f}"),
            ("" if r["ed_true_e_site"] is None else f"{r['ed_true_e_site']:.8f}"),
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


def write_overall_results(outdir: Path, rows_per_j2: List[Dict[str, Any]]):
    ensure_dir(outdir)

    csv_path = outdir / "overall_results.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("# Overall results: one row per J2\n")
        f.write("# Contains final E/site for: MLP(real/complex), RBM(real/complex) + deltas vs paper/ED where available.\n")
        f.write(
            "J2,"
            "MLP_real,MLP_real_err,MLP_complex,MLP_complex_err,"
            "RBM_real,RBM_real_err,RBM_complex,RBM_complex_err,"
            "paper_true,ed_true,"
            "dMLP_real_paper,dMLP_complex_paper,dRBM_real_paper,dRBM_complex_paper,"
            "dMLP_real_ed,dMLP_complex_ed,dRBM_real_ed,dRBM_complex_ed\n"
        )
        for r in rows_per_j2:
            def s(x, prec=12):
                return "" if x is None else f"{x:.{prec}f}"

            f.write(
                f"{r['J2']:.2f},"
                f"{s(r['MLP_real'])},{s(r['MLP_real_err'])},"
                f"{s(r['MLP_complex'])},{s(r['MLP_complex_err'])},"
                f"{s(r['RBM_real'])},{s(r['RBM_real_err'])},"
                f"{s(r['RBM_complex'])},{s(r['RBM_complex_err'])},"
                f"{s(r['paper_true'], prec=6)},{s(r['ed_true'])},"
                f"{s(r['dMLP_real_paper'])},{s(r['dMLP_complex_paper'])},{s(r['dRBM_real_paper'])},{s(r['dRBM_complex_paper'])},"
                f"{s(r['dMLP_real_ed'])},{s(r['dMLP_complex_ed'])},{s(r['dRBM_real_ed'])},{s(r['dRBM_complex_ed'])}\n"
            )

    headers = [
        "J2",
        "MLP_r", "±", "MLP_c", "±",
        "RBM_r", "±", "RBM_c", "±",
        "PAPER", "ED",
        "ΔMLP_r(P)", "ΔMLP_c(P)", "ΔRBM_r(P)", "ΔRBM_c(P)",
        "ΔMLP_r(ED)", "ΔMLP_c(ED)", "ΔRBM_r(ED)", "ΔRBM_c(ED)",
    ]

    def fmt(x, prec=6):
        return "" if x is None else f"{x:.{prec}f}"

    table = []
    for r in rows_per_j2:
        table.append([
            f"{r['J2']:.2f}",
            fmt(r["MLP_real"], 8), fmt(r["MLP_real_err"], 8),
            fmt(r["MLP_complex"], 8), fmt(r["MLP_complex_err"], 8),
            fmt(r["RBM_real"], 8), fmt(r["RBM_real_err"], 8),
            fmt(r["RBM_complex"], 8), fmt(r["RBM_complex_err"], 8),
            ("" if r["paper_true"] is None else f"{r['paper_true']:.5f}"),
            fmt(r["ed_true"], 8),
            fmt(r["dMLP_real_paper"], 8), fmt(r["dMLP_complex_paper"], 8),
            fmt(r["dRBM_real_paper"], 8), fmt(r["dRBM_complex_paper"], 8),
            fmt(r["dMLP_real_ed"], 8), fmt(r["dMLP_complex_ed"], 8),
            fmt(r["dRBM_real_ed"], 8), fmt(r["dRBM_complex_ed"], 8),
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
# New: mode helpers (train / aggregate)
# -----------------------------
def get_buckets(out_root: Path) -> Dict[str, Path]:
    buckets = {
        "MLP_complex": out_root / "MLP_complex",
        "MLP_real": out_root / "MLP_real",
        "RBM_complex": out_root / "RBM_complex",
        "RBM_real": out_root / "RBM_real",
        "MLP_vs_RBM_complex": out_root / "MLP_vs_RBM_complex",
        "MLP_vs_RBM_real": out_root / "MLP_vs_RBM_real",
        "MLP_real_vs_complex": out_root / "MLP_real_vs_complex",
        "RBM_real_vs_complex": out_root / "RBM_real_vs_complex",
        "compare_per_j2": out_root / "compare_per_j2",
    }
    for p in buckets.values():
        ensure_dir(p)
    return buckets


def load_run_from_disk(run_dir: Path, tag: str, J2: float, fallback_n_sites: int) -> Optional[Dict[str, np.ndarray]]:
    """
    Loads history + meta if available and returns a hist dict compatible with the in-memory one.
    """
    csv_p = history_csv_path(run_dir, tag, J2)
    if not csv_p.exists():
        return None

    hist = load_history_csv(csv_p)

    meta = load_json(run_dir / "run_meta.json") or {}
    n_sites = int(meta.get("n_sites", fallback_n_sites))
    runtime_s = float(meta.get("runtime_seconds", np.nan))
    n_params = int(meta.get("n_parameters", -1))

    hist["n_sites"] = np.array([n_sites])
    hist["runtime_s"] = np.array([runtime_s])
    hist["n_params"] = np.array([n_params])
    return hist


def build_summary_row(
    arch_name: str,
    dtype_str: str,
    J2: float,
    hist: Dict[str, np.ndarray],
    paper_true: Optional[float],
    ed_true: Optional[float],
) -> Dict[str, Any]:
    final_e = float(hist["e_site"][-1])
    final_err = float(hist["e_site_err"][-1])
    return {
        "arch": arch_name,
        "dtype": dtype_str,
        "J2": J2,
        "final_e_site": final_e,
        "final_e_site_err": final_err,
        "n_params": int(hist["n_params"][0]) if "n_params" in hist else -1,
        "runtime_s": float(hist["runtime_s"][0]) if "runtime_s" in hist else float("nan"),
        "paper_true_e_site": paper_true,
        "ed_true_e_site": ed_true,
    }


def main():
    parser = argparse.ArgumentParser(
        description="J1-J2 sweep with MLP/RBM and real/complex parameters, plus comparison plots."
    )

    # NEW
    parser.add_argument("--mode", type=str, default="sweep",
                        choices=["sweep", "train", "aggregate"],
                        help="sweep: original full run | train: run one config only | aggregate: merge/plot from existing outputs.")

    # NEW (train mode controls)
    parser.add_argument("--J2", type=float, default=None,
                        help="Single J2 value for --mode train. If set, overrides J2_list in train mode.")
    parser.add_argument("--arch", type=str, default=None, choices=[None, "mlp", "rbm"],
                        help="Architecture for --mode train.")
    parser.add_argument("--dtype", type=str, default=None, choices=[None, "real", "complex"],
                        help="Dtype for --mode train (real=float64, complex=complex128).")

    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--J1", type=float, default=1.0)
    parser.add_argument("--J2_list", type=str, default="0.4,0.5,0.6,1.0")

    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_iter", type=int, default=800)
    parser.add_argument("--discard", type=int, default=50)
    parser.add_argument("--diag_shift", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--mlp_hidden_scale", type=int, default=1)

    parser.add_argument("--rbm_lr", type=float, default=1e-3)
    parser.add_argument("--rbm_alpha", type=int, default=4)

    parser.add_argument("--platform", type=str, default=_platform,
                        choices=["auto", "cpu", "gpu", "tpu"])

    parser.add_argument("--out", type=str, default="results_j1j2_full_sweep")

    parser.add_argument("--ed_max_sites", type=int, default=20,
                        help="Compute NetKet ED reference only if N_sites <= ed_max_sites.")

    args = parser.parse_args()

    out_root = Path(args.out)
    ensure_dir(out_root)
    buckets = get_buckets(out_root)

    # For backward compatibility, parse the list always
    J2_list = parse_j2_list(args.J2_list)

    # Terminal header
    print("\n==============================")
    print("J1-J2 NQS Sweep (VMC_SR)")
    print("==============================")
    print(f"Mode:              {args.mode}")
    print(f"Requested platform:{args.platform}")
    print(f"JAX backend:       {jax.default_backend()}")
    print("JAX devices:       " + ", ".join([str(d) for d in jax.devices()]))
    print("------------------------------")
    print(f"L={args.L} -> N_sites={args.L * args.L}")
    print(f"J1={args.J1}")
    print(f"J2_list={J2_list}")
    print("------------------------------")
    print(f"n_samples={args.n_samples} | n_iter={args.n_iter} | discard={args.discard} | diag_shift={args.diag_shift}")
    print(f"MLP: lr={args.mlp_lr} | hidden_scale={args.mlp_hidden_scale}")
    print(f"RBM: lr={args.rbm_lr} | alpha={args.rbm_alpha}")
    print(f"ED reference: enabled only if N_sites <= {args.ed_max_sites}")
    print("==============================\n")

    # Save config (still useful for aggregate)
    save_json(out_root / "sweep_config.json", {
        "mode": args.mode,
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
        "ed_max_sites": args.ed_max_sites,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
    })

    # -----------------------------
    # MODE: TRAIN (single config)
    # -----------------------------
    if args.mode == "train":
        if args.J2 is None or args.arch is None or args.dtype is None:
            raise SystemExit("For --mode train you must provide: --J2 <float> --arch {mlp,rbm} --dtype {real,complex}")

        J2 = float(args.J2)
        paper_true = PAPER_TRUE_E_SITE.get(J2, None)
        ed_true = maybe_exact_ground_state_energy_per_site(args.L, args.J1, J2, args.ed_max_sites)

        dtype_str = args.dtype
        if dtype_str == "complex":
            dtype = jnp.complex128
        elif dtype_str == "real":
            dtype = jnp.float64
        else:
            raise ValueError("dtype must be 'real' or 'complex'")

        cfg = RunConfig(
            L=args.L, J1=args.J1, J2=J2,
            n_samples=args.n_samples,
            n_discard_per_chain=args.discard,
            n_iter=args.n_iter,
            diag_shift=args.diag_shift,
            seed=args.seed,
            mlp_hidden_scale=args.mlp_hidden_scale,
            rbm_alpha=args.rbm_alpha,
            mlp_lr=args.mlp_lr,
            rbm_lr=args.rbm_lr,
            param_dtype=dtype,
        )

        arch = args.arch.lower()
        tag = f"{arch.upper()}_{dtype_str}" if arch in ("mlp", "rbm") else f"{arch}_{dtype_str}"
        # Keep the same tag naming convention used previously:
        # MLP_complex / MLP_real / RBM_complex / RBM_real
        tag = f"{arch.upper()}_{dtype_str}".replace("MLP", "MLP").replace("RBM", "RBM")
        if arch == "mlp":
            outdir = buckets[f"MLP_{dtype_str}"] / f"J2_{J2:.2f}"
            print(f"[RUN] MLP ({dtype_str}) -> {outdir}")
            hist = run_single_vmc_sr("mlp", cfg, outdir, tag=f"MLP_{dtype_str}", lr=args.mlp_lr)

            plot_three_variants_energy_curve(
                outdir=outdir,
                base_name=f"MLP_{dtype_str}_energy_per_site",
                title_base=f"MLP ({dtype_str} params) | L={args.L} (N={args.L*args.L}) | J1={args.J1} | J2={J2} | samples={args.n_samples}",
                iters=hist["iters"],
                curves=[("MLP (E/site)", hist["e_site"], hist["e_site_err"])],
                paper_true=paper_true,
                ed_true=ed_true,
            )

            print("\n[RESULT] Final energy per site")
            print(f"  MLP ({dtype_str}): {fmt_pm(float(hist['e_site'][-1]), float(hist['e_site_err'][-1]))}")
            if paper_true is not None:
                print(f"  Paper true:        {paper_true:.6f}")
            if ed_true is not None:
                print(f"  NetKet ED true:    {ed_true:.12f}")

        elif arch == "rbm":
            outdir = buckets[f"RBM_{dtype_str}"] / f"J2_{J2:.2f}"
            print(f"[RUN] RBM ({dtype_str}) -> {outdir}")
            hist = run_single_vmc_sr("rbm", cfg, outdir, tag=f"RBM_{dtype_str}", lr=args.rbm_lr)

            plot_three_variants_energy_curve(
                outdir=outdir,
                base_name=f"RBM_{dtype_str}_energy_per_site",
                title_base=f"RBM ({dtype_str} params) | L={args.L} (N={args.L*args.L}) | J1={args.J1} | J2={J2} | samples={args.n_samples}",
                iters=hist["iters"],
                curves=[("RBM (E/site)", hist["e_site"], hist["e_site_err"])],
                paper_true=paper_true,
                ed_true=ed_true,
            )

            print("\n[RESULT] Final energy per site")
            print(f"  RBM ({dtype_str}): {fmt_pm(float(hist['e_site'][-1]), float(hist['e_site_err'][-1]))}")
            if paper_true is not None:
                print(f"  Paper true:        {paper_true:.6f}")
            if ed_true is not None:
                print(f"  NetKet ED true:    {ed_true:.12f}")

        print("\nDONE (train mode)")
        print(f"Outputs saved to: {out_root.resolve()}\n")
        return

    # -----------------------------
    # MODE: AGGREGATE (merge/plot-only)
    # -----------------------------
    if args.mode == "aggregate":
        print("[AGGREGATE] Reading existing runs from disk; no training will be performed.\n")

        results: Dict[Tuple[str, str, float], Dict[str, np.ndarray]] = {}
        summary_rows: List[Dict[str, Any]] = []
        n_sites_fallback = infer_n_sites_from_L(args.L)

        for J2 in J2_list:
            paper_true = PAPER_TRUE_E_SITE.get(J2, None)
            ed_true = maybe_exact_ground_state_energy_per_site(args.L, args.J1, J2, args.ed_max_sites)

            # Load all 4 configs if present
            for arch_name in ("MLP", "RBM"):
                for dtype_str in ("complex", "real"):
                    run_dir = buckets[f"{arch_name}_{dtype_str}"] / f"J2_{J2:.2f}"
                    tag = f"{arch_name}_{dtype_str}"
                    hist = load_run_from_disk(run_dir, tag=tag, J2=J2, fallback_n_sites=n_sites_fallback)
                    if hist is None:
                        print(f"[AGGREGATE] Missing: {arch_name} {dtype_str} J2={J2:.2f} -> {run_dir}")
                        continue

                    results[(arch_name, dtype_str, J2)] = hist
                    summary_rows.append(build_summary_row(
                        arch_name=arch_name,
                        dtype_str=dtype_str,
                        J2=J2,
                        hist=hist,
                        paper_true=paper_true,
                        ed_true=ed_true,
                    ))

                    # Re-generate per-model plots
                    if arch_name == "MLP":
                        plot_three_variants_energy_curve(
                            outdir=run_dir,
                            base_name=f"MLP_{dtype_str}_energy_per_site",
                            title_base=f"MLP ({dtype_str} params) | L={args.L} (N={args.L*args.L}) | J1={args.J1} | J2={J2} | samples={args.n_samples}",
                            iters=hist["iters"],
                            curves=[("MLP (E/site)", hist["e_site"], hist["e_site_err"])],
                            paper_true=paper_true,
                            ed_true=ed_true,
                        )
                    else:
                        plot_three_variants_energy_curve(
                            outdir=run_dir,
                            base_name=f"RBM_{dtype_str}_energy_per_site",
                            title_base=f"RBM ({dtype_str} params) | L={args.L} (N={args.L*args.L}) | J1={args.J1} | J2={J2} | samples={args.n_samples}",
                            iters=hist["iters"],
                            curves=[("RBM (E/site)", hist["e_site"], hist["e_site_err"])],
                            paper_true=paper_true,
                            ed_true=ed_true,
                        )

            # compare_per_j2 plots if the needed data exist
            have = all(
                (k in results)
                for k in [
                    ("MLP", "complex", J2),
                    ("RBM", "complex", J2),
                    ("MLP", "real", J2),
                    ("RBM", "real", J2),
                ]
            )
            if have:
                paper_true = PAPER_TRUE_E_SITE.get(J2, None)
                ed_true = maybe_exact_ground_state_energy_per_site(args.L, args.J1, J2, args.ed_max_sites)

                mlp_c = results[("MLP", "complex", J2)]
                rbm_c = results[("RBM", "complex", J2)]
                mlp_r = results[("MLP", "real", J2)]
                rbm_r = results[("RBM", "real", J2)]

                comp_dir = buckets["compare_per_j2"] / f"J2_{J2:.2f}"
                it = mlp_c["iters"]

                plot_three_variants_energy_curve(
                    outdir=comp_dir,
                    base_name=f"mlp_vs_rbm_real_J2_{J2:.2f}",
                    title_base=f"MLP vs RBM (real params) | L={args.L} (N={args.L*args.L}) | J1={args.J1} | J2={J2} | samples={args.n_samples}",
                    iters=it,
                    curves=[
                        ("MLP (real)", mlp_r["e_site"], mlp_r["e_site_err"]),
                        ("RBM (real)", rbm_r["e_site"], rbm_r["e_site_err"]),
                    ],
                    paper_true=paper_true,
                    ed_true=ed_true,
                )

                plot_three_variants_energy_curve(
                    outdir=comp_dir,
                    base_name=f"mlp_vs_rbm_complex_J2_{J2:.2f}",
                    title_base=f"MLP vs RBM (complex params) | L={args.L} (N={args.L*args.L}) | J1={args.J1} | J2={J2} | samples={args.n_samples}",
                    iters=it,
                    curves=[
                        ("MLP (complex)", mlp_c["e_site"], mlp_c["e_site_err"]),
                        ("RBM (complex)", rbm_c["e_site"], rbm_c["e_site_err"]),
                    ],
                    paper_true=paper_true,
                    ed_true=ed_true,
                )

                plot_three_variants_energy_curve(
                    outdir=comp_dir,
                    base_name=f"mlp_vs_rbm_all4_J2_{J2:.2f}",
                    title_base=f"MLP vs RBM (real+complex) | L={args.L} (N={args.L*args.L}) | J1={args.J1} | J2={J2} | samples={args.n_samples}",
                    iters=it,
                    curves=[
                        ("MLP (real)",    mlp_r["e_site"], mlp_r["e_site_err"]),
                        ("MLP (complex)", mlp_c["e_site"], mlp_c["e_site_err"]),
                        ("RBM (real)",    rbm_r["e_site"], rbm_r["e_site_err"]),
                        ("RBM (complex)", rbm_c["e_site"], rbm_c["e_site_err"]),
                    ],
                    paper_true=paper_true,
                    ed_true=ed_true,
                )
            else:
                print(f"[AGGREGATE] compare_per_j2 skipped for J2={J2:.2f} (missing one or more of the 4 runs).")

        # Write per-run summary (whatever exists)
        write_summary_files(out_root, summary_rows)

        # Overall results: one row per J2 (blank where missing)
        overall_rows: List[Dict[str, Any]] = []
        for J2 in J2_list:
            paper_true = PAPER_TRUE_E_SITE.get(J2, None)
            ed_true = maybe_exact_ground_state_energy_per_site(args.L, args.J1, J2, args.ed_max_sites)

            def get_final(arch: str, dtype_str: str) -> Tuple[Optional[float], Optional[float]]:
                k = (arch, dtype_str, J2)
                if k not in results:
                    return None, None
                h = results[k]
                return float(h["e_site"][-1]), float(h["e_site_err"][-1])

            MLP_real, MLP_real_err = get_final("MLP", "real")
            MLP_complex, MLP_complex_err = get_final("MLP", "complex")
            RBM_real, RBM_real_err = get_final("RBM", "real")
            RBM_complex, RBM_complex_err = get_final("RBM", "complex")

            def delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
                if a is None or b is None:
                    return None
                return float(a) - float(b)

            overall_rows.append({
                "J2": J2,
                "MLP_real": MLP_real, "MLP_real_err": MLP_real_err,
                "MLP_complex": MLP_complex, "MLP_complex_err": MLP_complex_err,
                "RBM_real": RBM_real, "RBM_real_err": RBM_real_err,
                "RBM_complex": RBM_complex, "RBM_complex_err": RBM_complex_err,
                "paper_true": paper_true,
                "ed_true": ed_true,
                "dMLP_real_paper": delta(MLP_real, paper_true),
                "dMLP_complex_paper": delta(MLP_complex, paper_true),
                "dRBM_real_paper": delta(RBM_real, paper_true),
                "dRBM_complex_paper": delta(RBM_complex, paper_true),
                "dMLP_real_ed": delta(MLP_real, ed_true),
                "dMLP_complex_ed": delta(MLP_complex, ed_true),
                "dRBM_real_ed": delta(RBM_real, ed_true),
                "dRBM_complex_ed": delta(RBM_complex, ed_true),
            })

        write_overall_results(out_root, overall_rows)

        print("\n==============================")
        print("DONE (aggregate mode)")
        print("==============================")
        print(f"Outputs saved to: {out_root.resolve()}")
        print("  - summary.csv / summary.txt")
        print("  - overall_results.csv / overall_results.txt")
        print("  - per-model and compare_per_j2 plots regenerated where data available.\n")
        return

    # -----------------------------
    # MODE: SWEEP (original behavior)
    # -----------------------------
    summary_rows: List[Dict[str, Any]] = []
    results: Dict[Tuple[str, str, float], Dict[str, np.ndarray]] = {}

    for J2 in J2_list:
        paper_true = PAPER_TRUE_E_SITE.get(J2, None)

        print(f"=== J2 = {J2:.2f} ===")
        print(f"Paper reference E/site: {paper_true if paper_true is not None else 'N/A'}")

        ed_true = maybe_exact_ground_state_energy_per_site(args.L, args.J1, J2, args.ed_max_sites)
        if ed_true is None:
            print("NetKet ED reference:    (skipped / not feasible for this size)")
        else:
            print(f"NetKet ED reference:    {ed_true:.12f}")

        for dtype_str, dtype in [("complex", jnp.complex128), ("real", jnp.float64)]:
            cfg = RunConfig(
                L=args.L, J1=args.J1, J2=J2,
                n_samples=args.n_samples,
                n_discard_per_chain=args.discard,
                n_iter=args.n_iter,
                diag_shift=args.diag_shift,
                seed=args.seed,
                mlp_hidden_scale=args.mlp_hidden_scale,
                rbm_alpha=args.rbm_alpha,
                mlp_lr=args.mlp_lr,
                rbm_lr=args.rbm_lr,
                param_dtype=dtype,
            )

            # MLP
            mlp_dir = buckets[f"MLP_{dtype_str}"] / f"J2_{J2:.2f}"
            print(f"\n[RUN] MLP ({dtype_str}) -> {mlp_dir}")
            mlp_hist = run_single_vmc_sr(
                arch="mlp", cfg=cfg, outdir=mlp_dir,
                tag=f"MLP_{dtype_str}", lr=args.mlp_lr
            )
            results[("MLP", dtype_str, J2)] = mlp_hist

            # RBM
            rbm_dir = buckets[f"RBM_{dtype_str}"] / f"J2_{J2:.2f}"
            print(f"[RUN] RBM ({dtype_str}) -> {rbm_dir}")
            rbm_hist = run_single_vmc_sr(
                arch="rbm", cfg=cfg, outdir=rbm_dir,
                tag=f"RBM_{dtype_str}", lr=args.rbm_lr
            )
            results[("RBM", dtype_str, J2)] = rbm_hist

            # Per-model plots
            plot_three_variants_energy_curve(
                outdir=mlp_dir,
                base_name=f"MLP_{dtype_str}_energy_per_site",
                title_base=f"MLP ({dtype_str} params) | L={args.L} (N={args.L*args.L}) | J1={args.J1} | J2={J2} | samples={args.n_samples}",
                iters=mlp_hist["iters"],
                curves=[("MLP (E/site)", mlp_hist["e_site"], mlp_hist["e_site_err"])],
                paper_true=paper_true,
                ed_true=ed_true,
            )
            plot_three_variants_energy_curve(
                outdir=rbm_dir,
                base_name=f"RBM_{dtype_str}_energy_per_site",
                title_base=f"RBM ({dtype_str} params) | L={args.L} (N={args.L*args.L}) | J1={args.J1} | J2={J2} | samples={args.n_samples}",
                iters=rbm_hist["iters"],
                curves=[("RBM (E/site)", rbm_hist["e_site"], rbm_hist["e_site_err"])],
                paper_true=paper_true,
                ed_true=ed_true,
            )

            # Summary rows (per run)
            for arch_name, hist in [("MLP", mlp_hist), ("RBM", rbm_hist)]:
                final_e = float(hist["e_site"][-1])
                final_err = float(hist["e_site_err"][-1])
                summary_rows.append({
                    "arch": arch_name,
                    "dtype": dtype_str,
                    "J2": J2,
                    "final_e_site": final_e,
                    "final_e_site_err": final_err,
                    "n_params": int(hist["n_params"][0]),
                    "runtime_s": float(hist["runtime_s"][0]),
                    "paper_true_e_site": paper_true,
                    "ed_true_e_site": ed_true,
                })

            print("\n[RESULT] Final energy per site")
            print(f"  MLP ({dtype_str}): {fmt_pm(float(mlp_hist['e_site'][-1]), float(mlp_hist['e_site_err'][-1]))}")
            print(f"  RBM ({dtype_str}): {fmt_pm(float(rbm_hist['e_site'][-1]), float(rbm_hist['e_site_err'][-1]))}")
            if paper_true is not None:
                print(f"  Paper true:        {paper_true:.6f}")
            if ed_true is not None:
                print(f"  NetKet ED true:    {ed_true:.12f}")
            print()

        # compare_per_j2 plots
        mlp_c = results[("MLP", "complex", J2)]
        rbm_c = results[("RBM", "complex", J2)]
        mlp_r = results[("MLP", "real", J2)]
        rbm_r = results[("RBM", "real", J2)]

        comp_dir = buckets["compare_per_j2"] / f"J2_{J2:.2f}"
        it = mlp_c["iters"]

        plot_three_variants_energy_curve(
            outdir=comp_dir,
            base_name=f"mlp_vs_rbm_real_J2_{J2:.2f}",
            title_base=f"MLP vs RBM (real params) | L={args.L} (N={args.L*args.L}) | J1={args.J1} | J2={J2} | samples={args.n_samples}",
            iters=it,
            curves=[
                ("MLP (real)", mlp_r["e_site"], mlp_r["e_site_err"]),
                ("RBM (real)", rbm_r["e_site"], rbm_r["e_site_err"]),
            ],
            paper_true=paper_true,
            ed_true=ed_true,
        )

        plot_three_variants_energy_curve(
            outdir=comp_dir,
            base_name=f"mlp_vs_rbm_complex_J2_{J2:.2f}",
            title_base=f"MLP vs RBM (complex params) | L={args.L} (N={args.L*args.L}) | J1={args.J1} | J2={J2} | samples={args.n_samples}",
            iters=it,
            curves=[
                ("MLP (complex)", mlp_c["e_site"], mlp_c["e_site_err"]),
                ("RBM (complex)", rbm_c["e_site"], rbm_c["e_site_err"]),
            ],
            paper_true=paper_true,
            ed_true=ed_true,
        )

        plot_three_variants_energy_curve(
            outdir=comp_dir,
            base_name=f"mlp_vs_rbm_all4_J2_{J2:.2f}",
            title_base=f"MLP vs RBM (real+complex) | L={args.L} (N={args.L*args.L}) | J1={args.J1} | J2={J2} | samples={args.n_samples}",
            iters=it,
            curves=[
                ("MLP (real)",    mlp_r["e_site"], mlp_r["e_site_err"]),
                ("MLP (complex)", mlp_c["e_site"], mlp_c["e_site_err"]),
                ("RBM (real)",    rbm_r["e_site"], rbm_r["e_site_err"]),
                ("RBM (complex)", rbm_c["e_site"], rbm_c["e_site_err"]),
            ],
            paper_true=paper_true,
            ed_true=ed_true,
        )

    # Write per-run summary
    write_summary_files(out_root, summary_rows)

    # Overall results
    overall_rows: List[Dict[str, Any]] = []
    for J2 in J2_list:
        paper_true = PAPER_TRUE_E_SITE.get(J2, None)
        ed_true = maybe_exact_ground_state_energy_per_site(args.L, args.J1, J2, args.ed_max_sites)

        mlp_r = results[("MLP", "real", J2)]
        mlp_c = results[("MLP", "complex", J2)]
        rbm_r = results[("RBM", "real", J2)]
        rbm_c = results[("RBM", "complex", J2)]

        MLP_real = float(mlp_r["e_site"][-1])
        MLP_real_err = float(mlp_r["e_site_err"][-1])
        MLP_complex = float(mlp_c["e_site"][-1])
        MLP_complex_err = float(mlp_c["e_site_err"][-1])

        RBM_real = float(rbm_r["e_site"][-1])
        RBM_real_err = float(rbm_r["e_site_err"][-1])
        RBM_complex = float(rbm_c["e_site"][-1])
        RBM_complex_err = float(rbm_c["e_site_err"][-1])

        def delta(a: float, b: Optional[float]) -> Optional[float]:
            return None if b is None else (a - float(b))

        overall_rows.append({
            "J2": J2,
            "MLP_real": MLP_real, "MLP_real_err": MLP_real_err,
            "MLP_complex": MLP_complex, "MLP_complex_err": MLP_complex_err,
            "RBM_real": RBM_real, "RBM_real_err": RBM_real_err,
            "RBM_complex": RBM_complex, "RBM_complex_err": RBM_complex_err,
            "paper_true": paper_true,
            "ed_true": ed_true,
            "dMLP_real_paper": delta(MLP_real, paper_true),
            "dMLP_complex_paper": delta(MLP_complex, paper_true),
            "dRBM_real_paper": delta(RBM_real, paper_true),
            "dRBM_complex_paper": delta(RBM_complex, paper_true),
            "dMLP_real_ed": delta(MLP_real, ed_true),
            "dMLP_complex_ed": delta(MLP_complex, ed_true),
            "dRBM_real_ed": delta(RBM_real, ed_true),
            "dRBM_complex_ed": delta(RBM_complex, ed_true),
        })

    write_overall_results(out_root, overall_rows)

    print("\n==============================")
    print("DONE")
    print("==============================")
    print(f"Outputs saved to: {out_root.resolve()}")
    print("Summary files:")
    print("  - summary.csv / summary.txt")
    print("  - overall_results.csv / overall_results.txt")
    print("Note: NetKet ED reference is skipped automatically unless N_sites <= ed_max_sites.\n")


if __name__ == "__main__":
    main()
