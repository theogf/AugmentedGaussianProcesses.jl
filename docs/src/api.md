# API Library

---
```@contents
Pages = ["api.md"]
```

```@meta
CurrentModule = AugmentedGaussianProcesses
```

## Module
```@docs
AugmentedGaussianProcesses
```

## Model Types

```@docs
GP
VGP
SVGP
```

## Likelihood Types

```@docs
GaussianLikelihood
StudentTLikelihood
LaplaceLikelihood
LogisticLikelihood
HeteroscedasticLikelihood
BayesianSVM
SoftMaxLikelihood
LogisticSoftMaxLikelihood
PoissonLikelihood
```

## Inference Types

```@docs
AnalyticVI
AnalyticSVI
GibbsSampling
QuadratureVI
QuadratureSVI
MCIntegrationVI
MCIntegrationSVI
```

## Functions and methods

```@docs
train!
predict_f
predict_y
proba_y
```

## Kernels

```@docs
RBFKernel
MaternKernel
```

## Kernel functions

```@docs
kernelmatrix
kernelmatrix!
getvariance
getlengthscales
```

## Prior Means

```@docs
ZeroMean
ConstantMean
EmpiricalMean
```

## Index

```@index
Pages = ["api.md"]
Module = ["AugmentedGaussianProcesses"]
Order = [:type, :function]
```
