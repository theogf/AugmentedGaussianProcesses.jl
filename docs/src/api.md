# Library

---

```@meta
CurrentModule = AugmentedGaussianProcesses
```

## Module
```@docs
AugmentedGaussianProcesses
```

## Types

```@docs
BatchGPRegression
SparseGPRegression
BatchStudentT
SparseStudentT
BatchXGPC
SparseXGPC
BatchBSVM
SparseBSVM
```

## Functions and methods

```@docs
train!(::OfflineGPModel)
regpredict(::GPModel,::AbstractArray)
regpredictproba
studenttpredict
studenttpredictproba
logitpredict
logitpredictproba
svmpredict
svmpredictproba
```

## Kernels

```@docs
RBFKernel
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
