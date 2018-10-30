# AugmentedGaussianProcesses!
[![Build Status](https://travis-ci.org/theogf/AugmentedGaussianProcesses.jl.svg?branch=master)](https://travis-ci.org/theogf/AugmentedGaussianProcesses.jl)
  <!--a href="https://theogf.github.io/AugmentedGaussianProcesses.jl/stable">
    <img src="https://img.shields.io/badge/docs-stable-blue.svg">
  </a-->
  <a href="https://theogf.github.io/AugmentedGaussianProcesses.jl/latest">
    <img src="https://img.shields.io/badge/docs-latest-blue.svg">
  </a>

AugmentedGaussianProcesses! is a Julia package in development for **Data Augmented Sparse Gaussian Processes**. It contains a collection of models for different **gaussian and non-gaussian likelihoods**, which are transformed via data augmentation into **conditionally conjugate likelihood** allowing for **extremely fast inference** via block coordinate updates.

# Packages models :
  - Two GP classifier models
    - **BSVM** : A Classifier with a likelihood equivalent to the classic SVM [IJulia example](https://github.com/theogf/AugmentedGaussianProcesses.jl/blob/master/examples/Classification%20-%20BSVM.ipynb)/[Reference][arxivbsvm]
    - **XGPC** : A Classifier with a Bernoulli likelihood with the logistic link [IJulia example](https://github.com/theogf/AugmentedGaussianProcesses.jl/blob/master/examples/Classification%20-%20XGPC.ipynb)/[Reference][arxivxgpc]
  - Two GP Regression models
    - **GPRegression** : The standard Gaussian Process regression model with a Gaussian Likelihood (no data augmentation was needed here) [IJulia example](https://github.com/theogf/AugmentedGaussianProcesses.jl/blob/master/examples/Classification%20-%20Gaussian.ipynb)/[Reference][arxivgpbigdata]
    - **StudentT** : The standard Gaussian Process regression with a Student-t likelihood (the degree of freedom ν is not optimizable for the moment) [IJulia example](https://github.com/theogf/AugmentedGaussianProcesses.jl/blob/master/examples/Classification%20-%20Gaussian.ipynb)/[Reference][jmlrstudentt]
  - More models in development
    - **MultiClass** : A multiclass classifier model, relying on a modified version of softmax
    - **Probit** : A Classifier with a Bernoulli likelihood with the probit link
    - **Online** : Allowing for all algorithms to work online as well

For each of these models you can either run the fullbatch or sparse version by adding the prefix `Batch` or `Sparse` to the model name.
## Install the package

The package requires Julia 1.0
Run in `Julia` press `]` and type `add AugmentedGaussianProcesses` (once it has been released), it will install the package and all its requirements

## Use the package

A complete documentation is currently being written, for now you can use this very basic example where `X_train` is a matrix ``N x D`` where `N` is the number of training points and `D` is the number of dimensions and `Y_train` is a vector of outputs.

```julia
using AugmentedGaussianProcesses
model = SparseXGPC(X_train,Y_train;Stochastic=true,batchsize=100,m=64,kernel=RBFKernel(1.0)) #Parameters after ; are optional
model.train(iterations=100)
Y_predic = model.predict(X_test) #For getting the label directly
Y_predic_prob = model.predictproba(X_test) #For getting the likelihood of predicting class 1
```
The documentation is currently worked on but I invite you to check the self explaining examples in the mean time [here](https://github.com/theogf/AugmentedGaussianProcesses.jl/blob/master/examples/)

## References :

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
