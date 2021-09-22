[![AugmentedGaussianProcesses.jl](assets/banner.png)](https://github.com/theogf/AugmentedGaussianProcesses.jl)

[![Docs Latest](https://img.shields.io/badge/docs-dev-blue.svg)](https://theogf.github.io/AugmentedGaussianProcesses.jl/dev)
[![Docs Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://theogf.github.io/AugmentedGaussianProcesses.jl/stable)
![BuildStatus](https://github.com/theogf/AugmentedGaussianProcesses.jl/workflows/CI/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/theogf/AugmentedGaussianProcesses.jl/badge.svg?branch=master)](https://coveralls.io/github/theogf/AugmentedGaussianProcesses.jl?branch=master)


A [Julia](http://julialang.org) package for Augmented and Normal Gaussian Processes.

***

### Author
- [Théo Galy-Fajou](https://theogf.github.io) PhD Student at Technical University of Berlin.

### Installation

AugmentedGaussianProcesses is a [registered package](http://pkg.julialang.org) and is symply installed by running
```julia
pkg> add AugmentedGaussianProcesses
```

### Basic example

Here is a simple example to start right away :
```julia
using AugmentedGaussianProcesses
model = SVGP(compose(SqExponentialKernel(), ScaleTransform(1.0)), LogisticLikelihood(), AnalyticVI(), inducingpoints(KmeansAlg(50), X_train))
train!(model, X_train, y_train; iterations=100)
y_pred = predict_y(model, X_test)
```

### Related Gaussian Processes packages

- [GaussianProcesses.jl](https://github.com/STOR-i/GaussianProcesses.jl) : General package for Gaussian Processes with many available likelihoods.
- [Stheno.jl](https://github.com/willtebbutt/Stheno.jl) : Package for Gaussian Process regression.
- [AbstractGPs.jl](https://github.com/JuliaGaussianProcesses/AbstractGPs.jl) : General package containing base functions for working with GPs.
- [GPLikelihoods.jl](https://github.com/JuliaGaussianProcesses/GPLikelihoods.jl) : Package to define likelihoods for latent GP models.
- [ApproximateGPs.jl](https://github.com/JuliaGaussianProcesses/ApproximateGPs.jl) : Package for variational GPs based on AbstractGPs.jl.
A general comparison between this package is done on [Julia GP Package Comparison](@ref).

### License

AugmentedGaussianProcesses.jl is licensed under the MIT "Expat" license; see
[LICENSE](https://github.com/theogf/AugmentedGaussianProcesses.jl/LICENSE.md) for
the full license text.
