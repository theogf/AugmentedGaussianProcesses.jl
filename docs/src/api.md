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
BayesianSVM
LogisticSoftMaxLikelihood
```

## Inference Types

```@docs
AnalyticVI
AnalyticSVI
GibbsSampling
QuadratureVI
QuadratureSVI
MCMCIntegrationVI
MCMCIntegrationSVI
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


## Index

```@index
Pages = ["api.md"]
Module = ["AugmentedGaussianProcesses"]
Order = [:type, :function]
```
