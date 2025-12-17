import numpy as np
import jax
import jax.numpy as jnp

import netket as nk
import netket.nn as nknn

from config import Config


def assert_gpu():
    backend = jax.default_backend()
    devices = jax.devices()
    print("JAX backend:", backend)
    print("JAX devices:", devices)
    if backend != "gpu":
        raise RuntimeError(
            "GPU NOT DETECTED. This run is configured to require GPU.\n"
            "Fix your environment (GPU jaxlib install + GPU allocation) and try again."
        )


def _activation_fn(name: str):
    name = (name or "none").lower()
    act_map = {
        "log_cosh": nknn.log_cosh,
        "silu": jax.nn.silu,
        "gelu": jax.nn.gelu,
        "none": None,
    }
    if name not in act_map:
        raise ValueError(f"Unknown activation '{name}'. Valid: {list(act_map.keys())}")
    return act_map[name]


def hidden_dims_from_arch(arch: str, n_sites: int):
    N = int(n_sites)
    arch_map = {
        "N": (N,),
        "N_N": (N, N),
        "N_N_N": (N, N, N),
        "N_N_N_N": (N, N, N, N),
    }
    if arch not in arch_map:
        raise ValueError(f"Unknown arch '{arch}'. Valid: {list(arch_map.keys())}")
    return arch_map[arch]


def build_system(cfg: Config):
    lattice = nk.graph.Square(length=cfg.L, max_neighbor_order=2, pbc=cfg.pbc)
    hilbert = nk.hilbert.Spin(s=0.5, total_sz=cfg.total_sz, N=lattice.n_nodes)
    H = nk.operator.Heisenberg(
        hilbert=hilbert,
        graph=lattice,
        J=[cfg.J1, cfg.J2],
        sign_rule=[False, False],
    )
    return lattice, hilbert, H


def build_model(cfg: Config, n_sites: int):
    act = _activation_fn(cfg.activation)
    hidden_dims = hidden_dims_from_arch(cfg.arch, n_sites)

    # IMPORTANT: we do NOT pass kernel_init/bias_init -> keep NetKet defaults
    model = nk.models.MLP(
        hidden_dims=hidden_dims,
        hidden_activations=act,
        output_activation=None,
        use_output_bias=cfg.use_output_bias,
        param_dtype=cfg.param_dtype,
    )
    return model


def _extract_history(log: nk.logging.RuntimeLog, key: str):
    """
    Extract a NetKet History object into numpy arrays safely.
    Returns dict with iters plus any available fields.
    """
    if key not in log.data:
        return None

    hist = log.data[key]
    out = {}

    # iters is usually present
    try:
        out["iters"] = np.asarray(hist.iters)
    except Exception:
        out["iters"] = None

    # Common NetKet stats: Mean/Sigma/Variance (not always all present)
    for field in ["Mean", "Sigma", "Variance", "R_hat", "TauCorr"]:
        try:
            val = getattr(hist, field)
            out[field] = np.asarray(val)
        except Exception:
            out[field] = None

    return out


def run_vmc(cfg: Config):
    """
    Runs VMC+SR and returns:
      - summary metrics
      - full optimization path (Energy Mean/Sigma/Variance over iters)
      - any other logged histories (if available)
    """
    assert_gpu()
    np.random.seed(cfg.seed)

    lattice, hilbert, H = build_system(cfg)
    model = build_model(cfg, lattice.n_nodes)

    sampler = nk.sampler.MetropolisExchange(
        hilbert=hilbert,
        graph=lattice,
        d_max=cfg.d_max,
        n_chains=cfg.n_chains,
    )

    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=int(cfg.n_samples),
        n_discard_per_chain=int(cfg.n_discard_per_chain),
        seed=int(cfg.seed),
    )

    opt = nk.optimizer.Adam(learning_rate=float(cfg.learning_rate))
    log = nk.logging.RuntimeLog()

    driver = nk.driver.VMC_SR(
        hamiltonian=H,
        optimizer=opt,
        diag_shift=float(cfg.diag_shift),
        variational_state=vstate,
    )

    try:
        driver.run(n_iter=int(cfg.n_iter), out=log)
    except KeyboardInterrupt:
        print("\n[Interrupted] Returning partial results.\n")

    # ---- Extract energy history (required) ----
    E = _extract_history(log, "Energy")
    if E is None or E["Mean"] is None or E["Sigma"] is None:
        raise RuntimeError("Energy history missing from log; run may have failed.")

    iters = E["iters"]
    energy_mean = np.asarray(E["Mean"]).real
    energy_sigma = np.asarray(E["Sigma"]).real
    energy_var = np.asarray(E["Variance"]).real if E["Variance"] is not None else None

    # score = tail mean of energy (robust metric)
    tail_start = int(0.8 * len(energy_mean))
    score = float(np.mean(energy_mean[tail_start:])) if len(energy_mean) else float("inf")

    # diagnostics (optional)
    rhat_last = None
    tauc_last = None
    if E.get("R_hat") is not None:
        try:
            rhat_last = float(np.asarray(E["R_hat"])[-1])
        except Exception:
            rhat_last = None
    if E.get("TauCorr") is not None:
        try:
            tauc_last = float(np.asarray(E["TauCorr"])[-1])
        except Exception:
            tauc_last = None

    # Also keep any other histories available (for “save everything”)
    all_histories = {}
    for k in list(log.data.keys()):
        hk = _extract_history(log, k)
        if hk is not None:
            all_histories[k] = hk

    return {
        # sizes
        "n_sites": int(lattice.n_nodes),
        "n_params": int(vstate.n_parameters),

        # primary history (ground-state path)
        "iters": iters,
        "energy_mean": energy_mean,
        "energy_sigma": energy_sigma,
        "energy_var": np.asarray(energy_var) if energy_var is not None else None,

        # summary
        "score": score,
        "final_E": float(energy_mean[-1]),
        "final_Eerr": float(energy_sigma[-1]),
        "per_site_E": float(energy_mean[-1] / lattice.n_nodes),
        "per_site_Eerr": float(energy_sigma[-1] / lattice.n_nodes),
        "min_E": float(np.min(energy_mean)),

        # diagnostics
        "rhat_last": rhat_last,
        "taucorr_last": tauc_last,

        # everything else logged (optional, but saved)
        "all_histories": all_histories,
    }
