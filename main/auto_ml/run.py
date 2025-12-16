import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn

import netket as nk
import netket.nn as nknn

from config import Config


def print_devices():
    # Useful to confirm GPU usage on the cluster
    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())


def _activation_fn(name: str):
    name = (name or "none").lower()
    act_map = {
        "log_cosh": nknn.log_cosh,
        "tanh": jnp.tanh,
        "relu": jax.nn.relu,
        "gelu": jax.nn.gelu,
        "silu": jax.nn.silu,
        "elu": jax.nn.elu,
        "none": None,
    }
    if name not in act_map:
        raise ValueError(f"Unknown activation '{name}'. Valid: {list(act_map.keys())}")
    return act_map[name]


def hidden_dims_from_arch(arch: str, n_sites: int):
    N = n_sites
    arch_map = {
        "N": (N,),
        "N_N": (N, N),
        "N_N_N": (N, N, N),
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

    kernel_init = nn.initializers.normal(stddev=float(cfg.init_std))
    bias_init = nn.initializers.normal(stddev=float(cfg.init_std))

    return nk.models.MLP(
        hidden_dims=hidden_dims,
        hidden_activations=act,
        output_activation=None,
        use_output_bias=cfg.use_output_bias,
        param_dtype=cfg.param_dtype,
        kernel_init=kernel_init,
        bias_init=bias_init,
    )


def build_sampler(cfg: Config, lattice, hilbert):
    if cfg.sampler_name != "exchange":
        raise ValueError("This project supports only sampler_name='exchange'.")
    return nk.sampler.MetropolisExchange(
        hilbert=hilbert,
        graph=lattice,
        d_max=cfg.d_max,
        n_chains=cfg.n_chains,
    )


def run_vmc(cfg: Config):
    """
    Runs VMC+SR and returns:
      - summary metrics
      - full energy history (iters, energy mean, sigma)
    """
    np.random.seed(cfg.seed)

    lattice, hilbert, H = build_system(cfg)
    model = build_model(cfg, lattice.n_nodes)
    sampler = build_sampler(cfg, lattice, hilbert)

    vstate = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=cfg.n_samples,
        n_discard_per_chain=cfg.n_discard_per_chain,
        seed=cfg.seed,
    )

    opt = nk.optimizer.Adam(learning_rate=cfg.learning_rate)
    log = nk.logging.RuntimeLog()

    driver = nk.driver.VMC_SR(
        hamiltonian=H,
        optimizer=opt,
        diag_shift=cfg.diag_shift,
        variational_state=vstate,
    )

    try:
        driver.run(n_iter=cfg.n_iter, out=log)
    except KeyboardInterrupt:
        print("\n[Interrupted] Returning partial results.\n")

    if "Energy" not in log.data:
        raise RuntimeError("No 'Energy' key found in log.data. Run may have failed early.")

    E_hist = log.data["Energy"]
    iters = np.asarray(E_hist.iters)
    energy = np.asarray(E_hist.Mean.real)
    err = np.asarray(E_hist.Sigma.real)

    tail_start = int(0.8 * len(energy))
    score = float(energy[tail_start:].mean()) if len(energy) else float("inf")

    def safe_last(field: str):
        try:
            arr = getattr(E_hist, field)
            return float(np.asarray(arr)[-1])
        except Exception:
            return None

    rhat_last = safe_last("R_hat")
    taucorr_last = safe_last("TauCorr")

    return {
        # identifiers / sizes
        "n_sites": int(lattice.n_nodes),
        "n_params": int(vstate.n_parameters),

        # history
        "iters": iters,
        "energy": energy,
        "err": err,

        # summary metrics
        "score": score,
        "final_E": float(energy[-1]),
        "final_Eerr": float(err[-1]),
        "per_site_E": float(energy[-1] / lattice.n_nodes),
        "per_site_Eerr": float(err[-1] / lattice.n_nodes),
        "min_E": float(energy.min()),

        # diagnostics (might be None)
        "rhat_last": rhat_last,
        "taucorr_last": taucorr_last,
    }


if __name__ == "__main__":
    print_devices()
    cfg = Config()
    res = run_vmc(cfg)
    print("Done. Score:", res["score"], "Final E:", res["final_E"])
