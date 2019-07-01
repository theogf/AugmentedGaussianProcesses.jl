# Julia GP Package Comparison

## AugmentedGaussianProcesses.jl vs [Stheno.jl](https://github.com/willtebbutt/Stheno.jl) vs [GaussianProcesses.jl](https://github.com/STOR-i/GaussianProcesses.jl)

There are already two other Gaussian Process packages in Julia, however their feature are quite orthogonal. They are roughly compared here:
AGP.jl stands for AugmentedGaussianProcesses.jl and GP.jl for GaussianProcesses.jl

# Likelihood

| Likelihood | AGP.jl | Stheno.jl | GP.jl |
| --- | :-: | :-: | :-: |
| Gaussian | ✓ | ✓ (multi-input/multi-output) | ✓ |
| Student-T | ✓ | ✖ | ✓ |
| Bernoulli | ✓ (Logistic) | ✖ | ✓ (Probit) |
| Bayesian-SVM  | ✓ | ✖ |  ✖ |
| Poisson  | ✖ | ✖ | ✓ |
| Exponential | ✖ | ✖ | ✓ |
| MultiClass | ✓ | ✖ | ✖ |

# Inference

| Inference | AGP.jl | Stheno.jl | GP.jl |
| --- | :-: | :-: | :-: |
| Analytic (Gaussian only) | ✓ | ✓ | ✓ |
| Variational Inference | ✓ (Analytic and Num. Appr.)  | ✖ | ✖ |
| Gibbs-Sampling | ✓  |   ✖  |   ✖  |
| MCMC |  ✖ | ✖  | ✓ |
| Expec. Propag.  |  ✖ | ✖  | ✖  |

# Kernels

| Kernel| AGP.jl | Stheno.jl | GP.jl |
| --- | :-: | :-: | :-: |
| RBF/Squared Exponential | ✓ | ✓ | ✓ |
| Matern  | ✓ | ✖ | ✓ |
| Const | ✖ | ✓ | ✓ |
| Linear | ✖ | ✓ | ✓  |
| Poly  | ✖ | ✓ | ✓ |
| Periodic  | ✖ | ✖ | ✓ |
| Exponentiated Quadratic  | ✖ | ✓ | ✖ |
| Rational Quadratic | ✖ | ✓ | ✓ |
| Wiener | ✖ | ✓ | ✖ |
| Sum of kernel | ✖ | ✖ | ✓ |
| Product of kernels | ✖ | ✖ | ✓ |

Note that the kernels will be defered to `MLKernels.jl` in the future.

# Other features

| Feature | AGP.jl | Stheno.jl | GP.jl |
| --- | :-: | :-: | :-: |
| Sparse GP | ✓ | ✖ | ✓  |
| Custom prior Mean | ✓ | ✓ | ✓ |
| Hyperparam. Opt. | ✓ | ? | ? |
| MultiOutput | ✖ | ✓ | ✖ |
