# AugmentedGaussianProcesses!
[![Docs Latest](https://img.shields.io/badge/docs-dev-blue.svg)](https://theogf.github.io/AugmentedGaussianProcesses.jl/dev)
[![Docs Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://theogf.github.io/AugmentedGaussianProcesses.jl/stable)
[![Build Status](https://travis-ci.org/theogf/AugmentedGaussianProcesses.jl.svg?branch=master)](https://travis-ci.org/theogf/AugmentedGaussianProcesses.jl)
[![Coverage Status](https://coveralls.io/repos/github/theogf/AugmentedGaussianProcesses.jl/badge.svg?branch=master)](https://coveralls.io/github/theogf/AugmentedGaussianProcesses.jl?branch=master)

AugmentedGaussianProcesses! (previously OMGP) is a Julia package in development for **Data Augmented Sparse Gaussian Processes**. It contains a collection of models for different **gaussian and non-gaussian likelihoods**, which are transformed via data augmentation into **conditionally conjugate likelihood** allowing for **extremely fast inference** via block coordinate updates.

# Packages models :

## Two GP classification likelihood
  - **BayesianSVM** : A Classifier with a likelihood equivalent to the classic SVM [IJulia example](https://github.com/theogf/AugmentedGaussianProcesses.jl/blob/master/examples/Classification%20-%20BSVM.ipynb)/[Reference][arxivbsvm]
  - **Logistic** : A Classifier with a Bernoulli likelihood with the logistic link [IJulia example](https://github.com/theogf/AugmentedGaussianProcesses.jl/blob/master/examples/Classification%20-%20XGPC.ipynb)/[Reference][arxivxgpc]

    ![Classification Plot](docs/figures/Classification.png)
---
## Two GP Regression likelihood
  - **Gaussian** : The standard Gaussian Process regression model with a Gaussian Likelihood (no data augmentation was needed here) [IJulia example](https://github.com/theogf/AugmentedGaussianProcesses.jl/blob/master/examples/Classification%20-%20Gaussian.ipynb)/[Reference][arxivgpbigdata]
  - **StudentT** : The standard Gaussian Process regression with a Student-t likelihood (the degree of freedom ν is not optimizable for the moment) [IJulia example](https://github.com/theogf/AugmentedGaussianProcesses.jl/blob/master/examples/Classification%20-%20Gaussian.ipynb)/[Reference][jmlrstudentt]

   ![Regression Plot](docs/figures/Regression.png)
---
## More models in development
  - **MultiClass** : A multiclass classifier model, relying on a modified version of softmax
  - **Heteroscedastic** : Non stationary noise
  - **Probit** : A Classifier with a Bernoulli likelihood with the probit link
  - **Online** : Allowing for all algorithms to work online as well
  - **Numerical solving** : Allow for a more general class of likelihoods by applying numerical solving

## Install the package

The package requires Julia 1.0
Run in `Julia` press `]` and type `add AugmentedGaussianProcesses`, it will install the package and all its requirements

## Use the package

A complete documentation is currently being written, for now you can use this very basic example where `X_train` is a matrix ``N x D`` where `N` is the number of training points and `D` is the number of dimensions and `Y_train` is a vector of outputs.

```julia
using AugmentedGaussianProcesses
model = SVGP(X_train,Y_train,RBFKernel(1.0),AugmentedLogisticLikelihood(),StochasticAnalyticInference(100),64)
train!(model,iterations=100)
Y_predic = predict_y(model,X_test) #For getting the label directly
Y_predic_prob = proba_y(model,X_test) #For getting the likelihood of predicting class 1
```
The documentation is currently worked on but I invite you to check the self explaining examples in the mean time [here](https://github.com/theogf/AugmentedGaussianProcesses.jl/blob/master/examples/)

## References :

Check out [my website for more news](https://theogf.github.io)

["Gaussian Processes for Machine Learning"](http://www.gaussianprocess.org/gpml/) by Carl Edward Rasmussen and Christopher K.I. Williams

ECML 17' "Bayesian Nonlinear Support Vector Machines for Big Data" by Florian Wenzel, Théo Galy-Fajou, Matthäus Deutsch and Marius Kloft. [https://arxiv.org/abs/1707.05532][arxivbsvm]

Arxiv "Efficient Gaussian Process Classification using Polya-Gamma Variables" by Florian Wenzel, Théo Galy-Fajou, Christian Donner, Marius Kloft and Manfred Opper. [https://arxiv.org/abs/1802.06383][arxivxgpc]

UAI 13' "Gaussian Process for Big Data" by James Hensman, Nicolo Fusi and Neil D. Lawrence [https://arxiv.org/abs/1309.6835][arxivgpbigdata]

JMLR 11' "Robust Gaussian process regression with a Student-t likelihood." by Jylänki Pasi, Jarno Vanhatalo, and Aki Vehtari.  [http://www.jmlr.org/papers/v12/jylanki11a.html][jmlrstudentt]

[arxivgpbigdata]:https://arxiv.org/abs/1309.6835
  [31b06e91]: https://github.com/theogf/AugmentedGaussianProcesses.jl/blob/master/examples/Classification%20-%20SXGPC.ipynb "Classification with Sparse XGPC"
[arxivbsvm]:https://arxiv.org/abs/1707.05532
[arxivxgpc]:https://arxiv.org/abs/1802.06383
[jmlrstudentt]:http://www.jmlr.org/papers/volume12/jylanki11a/jylanki11a.pdf
