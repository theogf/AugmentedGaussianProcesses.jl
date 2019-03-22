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
AugmentedStudentTLikelihood
BayesianSVM
AugmentedLogisticLikelihood
AugmentedLogisticSoftMaxLikelihood
```

## Inference Types

```@docs
AnalyticVI
AnalyticSVI
NumericalVI
NumericalSVI
QuadratureVI
QuadratureSVI
MCMCIntegrationVI
MCMCIntegrationSVI
GibbsSampling
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
