# AugmentedGaussianProcesses.jl
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://theogf.github.io/AugmentedGaussianProcesses.jl/stable)
A [Julia](http://julialang.org) package for Augmented and Normal Gaussian Processes.

---

### Authors
- [Théo Galy-Fajou](https://theogf.github.io) PhD Student at Technical University of Berlin.
- [Florian Wenzel](http://www.florian-wenzel.de) PhD Student at Technical University of Kaiserslautern and Humboldt University of Berlin

### Installation

AugmentedGaussianProcesses is a [registered package](http://pkg.julialang.org) and is symply installed by running
```julia
pkg> add AugmentedGaussianProcesses
```

### Basic example

Here is a simple example to start right away :
```julia
using AugmentedGaussianProcesses
model = SVGP(X_train,y_train,kernel=RBFKernel(1.0),AugmentedLogisticLikelihood(),AnalyticInference(),m=50)
train!(model,iterations=100)
y_pred = predict_y(model,X_test)
```

### Related Gaussian Processes packages

- [GaussianProcesses.jl](https://github.com/STOR-i/GaussianProcesses.jl) : General package for Gaussian Processes with many available likelihoods
- [Stheno.jl](https://github.com/willtebbutt/Stheno.jl) : Package for Gaussian Process regression

A general comparison between this package is done on [Julia GP Package Comparison](@ref). Benchmark evaluations may come later.

### License
AugmentedGaussianProcesses.jl is licensed under the MIT "Expat" license; see
[LICENSE](https://github.com/theogf/AugmentedGaussianProcesses.jl/LICENSE.md) for
the full license text.
