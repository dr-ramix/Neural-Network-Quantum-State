from dataclasses import dataclass
from typing import Optional, Any
import jax.numpy as jnp


@dataclass
class Config:
    # Physics
    L: int = 6
    J1: float = 1.0
    J2: float = 0.5
    pbc: bool = True
    total_sz: float = 0.0

    # Model
    arch: str = "N_N"
    activation: str = "log_cosh"
    init_std: float = 0.01
    use_output_bias: bool = True
    param_dtype: Any = jnp.complex128

    # Sampler / MC
    sampler_name: str = "exchange"
    d_max: int = 2
    n_chains: Optional[int] = None
    n_samples: int = 2048
    n_discard_per_chain: int = 10

    # Optimizer / SR
    learning_rate: float = 1e-3
    diag_shift: float = 1e-2

    # Run
    n_iter: int = 600
    seed: int = 1234
