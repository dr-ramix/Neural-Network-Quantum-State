from dataclasses import dataclass
from typing import Optional, Any, List
import jax.numpy as jnp


@dataclass
class Config:
    # Physics
    L: int = 6
    J1: float = 1.0
    J2: float = 0.5
    pbc: bool = True
    total_sz: float = 0.0

    # Model (NetKet nk.models.MLP)
    arch: str = "N_N"
    activation: str = "log_cosh"
    use_output_bias: bool = True
    param_dtype: Any = jnp.complex128  # keep consistent for paper

    # Sampler / MC
    d_max: int = 2
    n_chains: Optional[int] = None
    n_samples: int = 10_000  # <<< 10K sampling as requested
    n_discard_per_chain: int = 10

    # Optimizer / SR
    learning_rate: float = 3e-3
    diag_shift: float = 1e-2

    # Run
    n_iter: int = 400
    seed: int = 1234


# ---- Your grid (you can edit these) ----
ITERS: List[int]   = [400, 800]
SAMPLES: List[int] = [10_000]                 # <<< 10K only
LRS: List[float]   = [3e-3]
ARCHS: List[str]   = ["N", "N_N", "N_N_N", "N_N_N_N"]
ACTS: List[str]    = ["log_cosh", "silu", "gelu"]
SHIFTS: List[float]= [1e-2]
