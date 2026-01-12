#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Depth sweep for MLP Neural Quantum States (NetKet) on J1-J2 Heisenberg model:
  - Fixed: J1 = 1.0, J2 = 0.5
  - LxL square lattice (PBC), total_sz = 0 sector (as before)
  - MLP only, varying depth with width N = n_sites per hidden layer:
      depth=1 : (N)
      depth=2 : (N, N)
      depth=3 : (N, N, N)
      depth=4 : (N, N, N, N)
  - Run each depth once with real parameters and once with complex parameters.

Outputs (under --out):
  MLP_depth_real/
    depth_1/, depth_2/, depth_3/, depth_4/
  MLP_depth_complex/
    depth_1/, depth_2/, depth_3/, depth_4/
  compare_depth_per_dtype/
    real/   (all depths on one plot, 3 variants)
    complex/(all depths on one plot, 3 variants)
  compare_depth_real_vs_complex/
    (each depth: real vs complex, 3 variants)
  overall_results.csv / overall_results.txt
  per_run_summary.csv / per_run_summary.txt

For every plot:
  (a) no true value
  (b) with paper true value (E/site = -0.50381 at J2=0.5)
  (c) with NetKet ED true value if feasible (small systems only, controlled by --ed_max_sites)

GPU support:
  --platform gpu sets JAX_PLATFORM_NAME=gpu BEFORE importing jax/netket.
"""

import os
import json
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

# --- FIX 1: force headless matplotlib on clusters (before pyplot import) ---
import matplotlib
matplotlib.use("Agg")
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
# Paper "true" value provided by you
# -----------------------------
PAPER_TRUE_E_SITE = -0.50381  # for J2=0.5, energy per site


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

    # MLP optimizer
    mlp_lr: float = 1e-3

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


def build_mlp_model_depth(n_sites: int, depth: int, param_dtype):
    """
    Depth definition:
      depth=1 => hidden_dims=(N,)
      depth=2 => hidden_dims=(N,N)
      ...
    where N = n_sites
    """
    hidden_dims = tuple([n_sites] * depth)
    return nk.models.MLP(
        hidden_dims=hidden_dims,
        param_dtype=param_dtype,
        hidden_activations=log_cosh,
        output_activation=None,
        use_output_bias=True,
    )


# --- FIX 2: NetKet SR driver compatibility across versions ---
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


def maybe_exact_ground_state_energy_per_site(
    L: int,
    J1: float,
    J2: float,
    ed_max_sites: int,
) -> Optional[float]:
    """
    Tries NetKet ED/Lanczos for the ground state energy per site,
    only if N_sites <= ed_max_sites.

    Returns:
      energy_per_site (float) if computed else None
    """
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


def run_single_vmc_sr_mlp_depth(
    cfg: RunConfig,
    depth: int,
    outdir: Path,
    tag: str,
) -> Dict[str, np.ndarray]:
    ensure_dir(outdir)

    lattice, hilbert, ham = make_lattice_and_hamiltonian(cfg.L, cfg.J1, cfg.J2)
    n_sites = lattice.n_nodes
    sampler = build_sampler(hilbert, lattice, cfg.d_max)

    model = build_mlp_model_depth(n_sites=n_sites, depth=depth, param_dtype=cfg.param_dtype)
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

    iters = np.asarray(E_hist.iters, dtype=int)
    E_mean = np.asarray(E_hist.Mean.real, dtype=float)
    E_sigma = np.asarray(E_hist.Sigma.real, dtype=float)

    e_site = E_mean / n_sites
    e_site_err = E_sigma / n_sites

    # History CSV with metadata header
    csv_path = outdir / f"{tag}_history.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(f"# {tag} | MLP depth={depth} | hidden_dims=({','.join([str(n_sites)]*depth)})\n")
        f.write(f"# L={cfg.L} (N_sites={n_sites}) | J1={cfg.J1} | J2={cfg.J2}\n")
        f.write(f"# n_samples={cfg.n_samples} | n_iter={cfg.n_iter} | diag_shift={cfg.diag_shift} | seed={cfg.seed}\n")
        f.write("iter,energy_mean,energy_sigma,energy_per_site_mean,energy_per_site_sigma\n")
        for i, em, es, eps, epe in zip(iters, E_mean, E_sigma, e_site, e_site_err):
            f.write(f"{int(i)},{em:.12f},{es:.12f},{eps:.12f},{epe:.12f}\n")

    meta = {
        "tag": tag,
        "arch": "MLP",
        "depth": int(depth),
        "hidden_dims": [int(n_sites)] * int(depth),
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
        "n_sites": np.array([n_sites]),
        "runtime_s": np.array([t1 - t0]),
        "n_params": np.array([int(vstate.n_parameters)]),
    }


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


def write_per_run_summary(outdir: Path, rows: List[Dict[str, Any]]):
    """
    One row per run: (dtype, depth).
    """
    ensure_dir(outdir)

    csv_path = outdir / "per_run_summary.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("# Per-run summary (final energies per site)\n")
        f.write("dtype,depth,hidden_dims,n_params,runtime_s,final_e_site,final_e_site_err,paper_true_e_site,ed_true_e_site\n")
        for r in rows:
            # --- FIX 3: remove nested f-strings (Python syntax error) ---
            paper = "" if r["paper_true"] is None else f"{r['paper_true']:.6f}"
            ed = "" if r["ed_true"] is None else f"{r['ed_true']:.12f}"
            f.write(
                f"{r['dtype']},{r['depth']},\"{r['hidden_dims']}\",{r['n_params']},{r['runtime_s']:.6f},"
                f"{r['final_e_site']:.12f},{r['final_e_site_err']:.12f},{paper},{ed}\n"
            )

    headers = ["DTYPE", "DEPTH", "HIDDEN_DIMS", "N_PARAMS", "RUNTIME(s)", "FINAL E/SITE", "ERR", "PAPER", "ED"]
    table = []
    for r in rows:
        table.append([
            r["dtype"],
            str(r["depth"]),
            str(r["hidden_dims"]),
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
    One row per depth, side-by-side real vs complex.
    """
    ensure_dir(outdir)

    csv_path = outdir / "overall_results.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("# Overall results: one row per depth, real vs complex side-by-side\n")
        f.write("depth,hidden_dims,paper_true,ed_true,real_e,real_err,complex_e,complex_err,d_real_paper,d_complex_paper,d_real_ed,d_complex_ed\n")
        for r in rows:
            def s(x, prec=12):
                return "" if x is None else f"{x:.{prec}f}"
            paper = "" if r["paper_true"] is None else f"{r['paper_true']:.6f}"
            f.write(
                f"{r['depth']},\"{r['hidden_dims']}\",{paper},"
                f"{s(r['ed_true'])},"
                f"{s(r['real_e'])},{s(r['real_err'])},"
                f"{s(r['complex_e'])},{s(r['complex_err'])},"
                f"{s(r['d_real_paper'])},{s(r['d_complex_paper'])},"
                f"{s(r['d_real_ed'])},{s(r['d_complex_ed'])}\n"
            )

    headers = ["DEPTH", "HIDDEN_DIMS", "PAPER", "ED", "REAL E", "±", "CPLX E", "±", "ΔREAL(P)", "ΔCPLX(P)", "ΔREAL(ED)", "ΔCPLX(ED)"]
    def fmt(x, prec=8):
        return "" if x is None else f"{x:.{prec}f}"

    table = []
    for r in rows:
        table.append([
            str(r["depth"]),
            str(r["hidden_dims"]),
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


def main():
    parser = argparse.ArgumentParser(
        description="Depth sweep for MLP NQS on J1-J2 Heisenberg (fixed J1=1, J2=0.5)."
    )

    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--J1", type=float, default=1.0)
    parser.add_argument("--J2", type=float, default=0.5)

    parser.add_argument("--depths", type=str, default="1,2,3,4",
                        help="Comma-separated depths to run (e.g. 1,2,3,4).")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_iter", type=int, default=600)
    parser.add_argument("--discard", type=int, default=50)
    parser.add_argument("--diag_shift", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--d_max", type=int, default=2)

    parser.add_argument("--platform", type=str, default=_platform,
                        choices=["auto", "cpu", "gpu", "tpu"])
    parser.add_argument("--out", type=str, default="results_mlp_depth_sweep_J2_0p5")

    parser.add_argument("--ed_max_sites", type=int, default=20,
                        help="Compute NetKet ED reference only if N_sites <= ed_max_sites.")

    args = parser.parse_args()

    depths = [int(x.strip()) for x in args.depths.split(",") if x.strip()]
    out_root = Path(args.out)
    ensure_dir(out_root)

    # Fixed “paper true” is for J2=0.5; if user changes J2, we still plot paper line only if it matches 0.5.
    paper_true = PAPER_TRUE_E_SITE if abs(args.J2 - 0.5) < 1e-12 else None

    # ED reference (only one, since J1/J2 fixed)
    ed_true = maybe_exact_ground_state_energy_per_site(args.L, args.J1, args.J2, args.ed_max_sites)

    # Terminal header
    n_sites = args.L * args.L
    print("\n===================================================")
    print("MLP Depth Sweep (VMC_SR) — Fixed Couplings")
    print("===================================================")
    print(f"Requested platform: {args.platform}")
    print(f"JAX backend:        {jax.default_backend()}")
    print("JAX devices:        " + ", ".join([str(d) for d in jax.devices()]))
    print("---------------------------------------------------")
    print(f"L={args.L} -> N_sites={n_sites}")
    print(f"J1={args.J1} | J2={args.J2}")
    print(f"Depths={depths}  (each hidden layer width = N_sites)")
    print("---------------------------------------------------")
    print(f"n_samples={args.n_samples} | n_iter={args.n_iter} | discard={args.discard} | diag_shift={args.diag_shift} | seed={args.seed}")
    print(f"MLP Adam lr={args.mlp_lr} | sampler=MetropolisExchange(d_max={args.d_max})")
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
    save_json(out_root / "config.json", {
        "L": args.L,
        "J1": args.J1,
        "J2": args.J2,
        "depths": depths,
        "n_samples": args.n_samples,
        "n_iter": args.n_iter,
        "discard": args.discard,
        "diag_shift": args.diag_shift,
        "seed": args.seed,
        "mlp_lr": args.mlp_lr,
        "d_max": args.d_max,
        "paper_true_e_site": paper_true,
        "ed_true_e_site": ed_true,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
    })

    # Output dirs
    real_root = out_root / "MLP_depth_real"
    cplx_root = out_root / "MLP_depth_complex"
    cmp_dtype_root = out_root / "compare_depth_per_dtype"
    cmp_rvc_root = out_root / "compare_depth_real_vs_complex"
    for p in [real_root, cplx_root, cmp_dtype_root, cmp_rvc_root]:
        ensure_dir(p)

    # Results store:
    # hist[(dtype_str, depth)] = hist
    hist: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}

    per_run_rows: List[Dict[str, Any]] = []

    # Run all depths for both dtypes
    for dtype_str, dtype in [("real", jnp.float64), ("complex", jnp.complex128)]:
        print(f"\n#############################")
        print(f"### Running dtype = {dtype_str.upper()}")
        print(f"#############################\n")

        for depth in depths:
            cfg = RunConfig(
                L=args.L, J1=args.J1, J2=args.J2,
                n_samples=args.n_samples,
                n_discard_per_chain=args.discard,
                n_iter=args.n_iter,
                diag_shift=args.diag_shift,
                seed=args.seed,
                d_max=args.d_max,
                mlp_lr=args.mlp_lr,
                param_dtype=dtype,
            )

            outdir = (real_root if dtype_str == "real" else cplx_root) / f"depth_{depth}"
            tag = f"MLP_{dtype_str}_depth_{depth}"

            hidden_dims = [n_sites] * depth
            print(f"[RUN] {tag}")
            print(f"      hidden_dims={hidden_dims}")
            print(f"      outdir={outdir}")

            h = run_single_vmc_sr_mlp_depth(cfg=cfg, depth=depth, outdir=outdir, tag=tag)
            hist[(dtype_str, depth)] = h

            final_e = float(h["e_site"][-1])
            final_err = float(h["e_site_err"][-1])
            print(f"[DONE] Final E/site: {fmt_pm(final_e, final_err)}")
            if paper_true is not None:
                print(f"       Paper true:   {paper_true:.6f}  (Δ={final_e - paper_true:+.6f})")
            if ed_true is not None:
                print(f"       NetKet ED:    {ed_true:.12f}  (Δ={final_e - ed_true:+.6f})")
            print()

            # Per-run plot (3 variants)
            title_base = (
                f"MLP depth={depth} ({dtype_str} params) | hidden_dims={tuple(hidden_dims)} | "
                f"L={args.L} (N_sites={n_sites}) | J1={args.J1} | J2={args.J2} | samples={args.n_samples}"
            )
            plot_three_variants_energy_curve(
                outdir=outdir,
                base_name=f"{tag}_energy_per_site",
                title_base=title_base,
                iters=h["iters"],
                curves=[(f"depth={depth}", h["e_site"], h["e_site_err"])],
                paper_true=paper_true,
                ed_true=ed_true,
            )

            per_run_rows.append({
                "dtype": dtype_str,
                "depth": depth,
                "hidden_dims": hidden_dims,
                "n_params": int(h["n_params"][0]),
                "runtime_s": float(h["runtime_s"][0]),
                "final_e_site": final_e,
                "final_e_site_err": final_err,
                "paper_true": paper_true,
                "ed_true": ed_true,
            })

    # Write per-run summary
    write_per_run_summary(out_root, per_run_rows)

    # -----------------------------------------
    # Comparisons: depth vs depth (per dtype)
    #   - real: all depths on one plot (3 variants)
    #   - complex: all depths on one plot (3 variants)
    # -----------------------------------------
    for dtype_str in ["real", "complex"]:
        curves = []
        iters_ref = None
        for depth in depths:
            h = hist[(dtype_str, depth)]
            if iters_ref is None:
                iters_ref = h["iters"]
            curves.append((f"depth={depth}", h["e_site"], h["e_site_err"]))

        outdir = cmp_dtype_root / dtype_str
        ensure_dir(outdir)

        title_base = (
            f"MLP depth comparison ({dtype_str} params) | L={args.L} (N_sites={n_sites}) | "
            f"J1={args.J1} | J2={args.J2} | samples={args.n_samples}"
        )
        plot_three_variants_energy_curve(
            outdir=outdir,
            base_name=f"compare_depths_{dtype_str}",
            title_base=title_base,
            iters=iters_ref,
            curves=curves,
            paper_true=paper_true,
            ed_true=ed_true,
        )

    # -----------------------------------------
    # Comparisons: real vs complex (per depth)
    #   For each depth, plot real vs complex (3 variants)
    # -----------------------------------------
    for depth in depths:
        h_r = hist[("real", depth)]
        h_c = hist[("complex", depth)]
        outdir = cmp_rvc_root / f"depth_{depth}"
        ensure_dir(outdir)

        title_base = (
            f"MLP real vs complex | depth={depth} | hidden_dims={(n_sites,)*depth} | "
            f"L={args.L} (N_sites={n_sites}) | J1={args.J1} | J2={args.J2} | samples={args.n_samples}"
        )

        plot_three_variants_energy_curve(
            outdir=outdir,
            base_name=f"compare_real_vs_complex_depth_{depth}",
            title_base=title_base,
            iters=h_r["iters"],
            curves=[
                ("real params", h_r["e_site"], h_r["e_site_err"]),
                ("complex params", h_c["e_site"], h_c["e_site_err"]),
            ],
            paper_true=paper_true,
            ed_true=ed_true,
        )

    # -----------------------------------------
    # One plot comparing ALL (depth,dtype) together (8 curves) (3 variants)
    # -----------------------------------------
    outdir_all = out_root / "compare_all_depths_and_dtypes"
    ensure_dir(outdir_all)

    curves_all = []
    iters_ref = hist[("real", depths[0])]["iters"]
    for dtype_str in ["real", "complex"]:
        for depth in depths:
            h = hist[(dtype_str, depth)]
            curves_all.append((f"{dtype_str}, depth={depth}", h["e_site"], h["e_site_err"]))

    title_base_all = (
        f"MLP depth sweep: all depths + real/complex | L={args.L} (N_sites={n_sites}) | "
        f"J1={args.J1} | J2={args.J2} | samples={args.n_samples}"
    )
    plot_three_variants_energy_curve(
        outdir=outdir_all,
        base_name="compare_all_depths_and_dtypes",
        title_base=title_base_all,
        iters=iters_ref,
        curves=curves_all,
        paper_true=paper_true,
        ed_true=ed_true,
    )

    # -----------------------------------------
    # Overall results (one row per depth, real vs complex side-by-side)
    # -----------------------------------------
    overall_rows = []
    for depth in depths:
        h_r = hist[("real", depth)]
        h_c = hist[("complex", depth)]

        real_e = float(h_r["e_site"][-1])
        real_err = float(h_r["e_site_err"][-1])
        cplx_e = float(h_c["e_site"][-1])
        cplx_err = float(h_c["e_site_err"][-1])

        def delta(a: float, b: Optional[float]) -> Optional[float]:
            return None if b is None else a - float(b)

        overall_rows.append({
            "depth": depth,
            "hidden_dims": [n_sites] * depth,
            "paper_true": paper_true,
            "ed_true": ed_true,
            "real_e": real_e, "real_err": real_err,
            "complex_e": cplx_e, "complex_err": cplx_err,
            "d_real_paper": delta(real_e, paper_true),
            "d_complex_paper": delta(cplx_e, paper_true),
            "d_real_ed": delta(real_e, ed_true),
            "d_complex_ed": delta(cplx_e, ed_true),
        })

    write_overall_results(out_root, overall_rows)

    print("\n===================================================")
    print("DONE")
    print("===================================================")
    print(f"All outputs saved to: {out_root.resolve()}")
    print("Summary files:")
    print("  - per_run_summary.csv / per_run_summary.txt   (one row per run)")
    print("  - overall_results.csv / overall_results.txt   (one row per depth, real vs complex)")
    print("Plots:")
    print("  - per-run (each depth, each dtype) in MLP_depth_* folders")
    print("  - depth comparisons per dtype in compare_depth_per_dtype/")
    print("  - real vs complex per depth in compare_depth_real_vs_complex/")
    print("  - all curves together in compare_all_depths_and_dtypes/")
    print("Note: NetKet ED reference is skipped unless N_sites <= ed_max_sites.\n")


if __name__ == "__main__":
    main()
