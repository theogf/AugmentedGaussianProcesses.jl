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
MCGP
SVGP
OnlineSVGP
MOVGP
MOSVGP
VStP
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
NegBinomialLikelihood
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
sample
predict_f
predict_y
proba_y
```

## Prior Means

```@docs
ZeroMean
ConstantMean
EmpiricalMean
AffineMean
```

## Index

```@index
Pages = ["api.md"]
Module = ["AugmentedGaussianProcesses"]
Order = [:type, :function]
```
