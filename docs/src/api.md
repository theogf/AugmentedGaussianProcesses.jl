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
GP(X,y,kernel)
VGP(X,y,kernel,likelihood,inference)
SVGP(X,y,kernel,likelihood,inference,num_ind_points)
```

## Likelihood Types

```@docs
GaussianLikelihood()
AugmentedStudentTLikelihood(ν)
BayesianSVM()
AugmentedLogisticLikelihood()
AugmentedLogisticSoftMaxLikelihood()
```

## Inference Types

```@docs
AnalyticInference()
StochasticAnalyticInference(batchsize)
NumericalInference(optimizer)
StochasticNumericalInference(batchsize)
GibbsSampling()
```

## Functions and methods

```@docs
train!(model)
predict_f(model,X_test)
predict_y(model,X_test)
proba_y(model,X_test)
```

## Kernels

```@docs
RBFKernel()
MaternKernel)_
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
