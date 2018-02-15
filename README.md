# OMGP!

Oh My GP! is a Julia package in development for **extremely efficient Gaussian Processes algorithms**. It contains for the moment only two classifiers : the Bayesian SVM, and a state-of-the-art algorithm for classification using the logit link called X-GPC. It is planned to implement Regression, as well as more complex likelihood, including the multi-class classification.

## Install the package

Run in Julia `Pkg.clone("github.com/theogf/OMGP.jl")`, it will install the package and all its requirements


## Use the package

```
using OMGPC
model = SparseXGPC(X_train,Y_train;Stochastic=?,BatchSize=?,m=?,Kernels=?) #Parameters after ; are optional
model.train(iterations=100)
Y_predic = model.predict(X_test) #For getting the label directly
Y_predic_prob = model.predictproba(X_test) #For getting the likelihood of predicting class 1
```
