# AugmentedGaussianProcesses.jl
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://theogf.github.io/AugmentedGaussianProcesses.jl/stable)
A [Julia](http://julialang.org) package for Augmented and Normal Gaussian Processes.

---

### Authors
- [Theo Galy-Fajou](https://github.com/theogf) PhD Student at Technical University of Berlin.
- [Florian Wenzel](http://www.florian-wenzel.de) PhD Student at Technical University of Kaiserslautern

### License
AugmentedGaussianProcesses.jl is licensed under the MIT "Expat" license; see
[LICENSE](https://github.com/theogf/AugmentedGaussianProcesses.jl/LICENSE.md) for
the full license text.


### Installation

AugmentedGaussianProcesses is a [registered package](http://pkg.julialang.org) and is symply installed by running
```julia
pkg> add AugmentedGaussianProcesses
```

### Basic example

Here is a simple example to start right away :
```julia
using AugmentedGaussianProcesses
model = SparseXGPC(X_train,y_train,kernel=RBFKernel(1.0),m=50)
model.train(iterations=100)
y_pred = model.predict(X_test)
```

### Related Gaussian Processes packages

- [GaussianProcesses.jl](https://github.com/STOR-i/GaussianProcesses.jl) : General package for Gaussian Processes with many available likelihoods
- [Stheno.jl](https://github.com/willtebbutt/Stheno.jl) : Package for Gaussian Process regression
