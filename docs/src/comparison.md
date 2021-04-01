# Julia GP Package Comparison

## JuliaGaussianProcesses Organization

There is a common effort to bring the GP people together through the JuliaGP organization.
We work on making the building blocks necessary for GP such as [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl) for kernels, [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl) for the basic GP definitions and more is coming.
The long-term goal is to have AGP.jl depend on all of this elements and to use it as a wrapper.

## 🚧 This comparison is now quite outdated and new solutions have been introduced 🚧 
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
| Poisson  | ✓ | ✖ | ✓ |
| NegativeBinomial  | ✓ | ✖ | ✖ |
| Exponential | ✖ | ✖ | ✓ |
| MultiClass | ✓ | ✖ | ✖ |

# Inference

| Inference | AGP.jl | Stheno.jl | GP.jl |
| --- | :-: | :-: | :-: |
| Analytic (Gaussian only) | ✓ | ✓ | ✓ |
| Variational Inference | ✓ (Analytic and Num. Appr.) | ✖ | ✖ |
| Streaming VI | ✓ | ✖ | ✖ |
| Gibbs-Sampling | ✓ | ✖ | ✖ |
| MCMC | ✖ | ✖ | ✓ |
| Expec. Propag. | ✖ | ✖ | ✖ |

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
| Hyperparam. Opt. | ✓ | ? | ✓ |
| MultiOutput | ✓ | ✓ | ✖ |
| Online  | ✓ | ✖ | ✖ |
