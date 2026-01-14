# Ground-State Estimation with Neural Quantum States
### Hands-on Examples for the J1–J2 Heisenberg Model

This repository contains a research-oriented codebase for studying **Neural Quantum States (NQS)** applied to the **spin-1/2 J1–J2 Heisenberg model**. It utilizes **NetKet**, **JAX**, and **Variational Monte Carlo (VMC)** methods to approximate ground states.

The project is designed to bridge **theoretical physics** and **practical implementation**, demonstrating how concepts from quantum many-body physics translate into modern machine-learning-based variational ansätze.


## Repository Structure

The core of the project is located in the `main/` directory, organized into **Theory** (educational resources) and **Experiments** (numerical benchmarks).

```text
nqs/
├── main/
│   ├── theory/             # Conceptual Notebooks (Bridge between Physics & Code)
│   │   ├── mlp/            # Multilayer Perceptron implementations
│   │   └── rbm/            # Restricted Boltzmann Machine implementations
│   │
│   ├── experiments/        # Reproducible Numerical Studies
│   │   ├── rbm_vs_mlp/     # Architecture comparisons
│   │   ├── complex_vs_real/# Parameter arithmetic studies
│   │   ├── model_depth/    # Impact of network depth
│   │   ├── activations/    # Activation function benchmarks
│   │   └── optimizers/     # Adam vs AdamW comparisons
│   │
│   └── [legacy folders]    # Side-experiments
│
├── requirements.txt        # Project dependencies
└── venv/                   # Local virtual environment
```

1. Theory & Concepts (main/theory)
This module provides a "physics-first" introduction to Neural Quantum States.
It contains Jupyter notebooks designed to bridge abstract mathematical theory with concrete NetKet/JAX implementation.

Key Implementations:
MLP: Feed-forward neural networks as wave functions.
RBM: Restricted Boltzmann Machines.
Topics CoveredHilbert Space: Mapping the physical model to the code.Variational
Ansätze: Mathematical formulation of the wave function $\psi_\theta$.
The NetKet Ecosystem: Understanding Samplers, Operators, and Variational States.
Stochastic Reconfiguration (SR): Optimization techniques for energy minimization.



2. Numerical Experiments (main/experiments)This directory contains systematic, reproducible studies.
   Each subdirectory focuses on a specific hyperparameter or architectural comparison.

Experimental SetupPhysical:
Model: Spin-1/2 J1–J2 Heisenberg model on a square lattice.
Hamiltonian: $J_1 = 1.0$ (fixed), with varying $J_2$ (e.g., $0.4, 0.5, 0.6, 1.0$).
Symmetry: Zero magnetization sector ($S^z_{tot} = 0$).
Benchmark CategoriesRBM vs. MLP: Comparing the expressivity of different architectures.
Complex vs. Real Parameters: Analyzing the necessity of complex weights for ground-state estimation.Model 
Depth: Impact of adding layers to the ansatz.
Activation Functions: Performance of different non-linearities (e.g., LogCosh, ReLU, GeLU).
Optimizers: Comparative study of Adam vs AdamW in the context of VMC.3. Results & Output FormatThe code prioritizes reproducibility.

Experiment scripts generate structured output directories containing metadata, logs, and summary tables.
Optimization History (history.csv)
Each run tracks energy metrics per iteration.

Code-Snippet
# MLP_complex history | L=6 (N=36) | J1=1.0 | J2=0.5
# n_samples=10000 | n_iter=800 | diag_shift=0.01 | seed=1234
iter,energy_mean,energy_sigma,energy_per_site_mean,energy_per_site_sigma
0,97.950538,0.283818,2.720848,0.007883
1,97.623274,0.282013,2.711757,0.007833
2,97.288895,0.299014,2.702469,0.008305
...

Aggregated Summaries (overall_results.txt)
Results are compiled to compare architectures against literature values ("PAPER") or Exact Diagonalization ("ED"). Below is a summary of the energy outcomes for varying $J_2$ values:
J2 = 0.40: MLP_r (-0.8213 ± 0.0027) vs RBM_c (-1.8862 ± 0.0017). Paper Ref: -0.52975
J2 = 0.50: MLP_r (-0.6675 ± 0.0046) vs RBM_c (-1.7260 ± 0.0028). Paper Ref: -0.50381
J2 = 0.60: MLP_r (-0.7896 ± 0.0050) vs RBM_c (-1.7443 ± 0.0016). Paper Ref: -0.49518
J2 = 1.00: MLP_r (-1.0071 ± 0.0055) vs RBM_c (-2.7114 ± 0.0021). Paper Ref: -0.714364



Installation & UsageIt is recommended to use the provided virtual environment setup.SetupBash

# 1. Create a virtual environment
    python -m venv venv

# 2. Activate the environment
    source venv/bin/activate

# 3. Install dependencies
    pip install -r requirements.txt

# Running an ExperimentNavigate to an experiment folder and execute the script:
    cd main/experiments/rbm_vs_mlp
    python script.py
