# Library

---

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
AnalyticInference
StochasticAnalyticInference
NumericalInference
StochasticNumericalInference
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
