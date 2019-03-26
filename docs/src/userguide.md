# User Guide

There are 3 main stages for the GPs:

- [Initialization](@ref init)
- [Training](@ref train)
- [Prediction](@ref pred)

## [Initialization](@id init)

### GP vs VGP vs SVGP

- `GP` corresponds to the original GP regression model
- `VGP` is a variational GP model, a multivariate Gaussian is approximating the true posterior. There is no inducing points augmentation involved. Therefore it is fitted for small datasets (~10^3 samples)
- `SVGP` is a variational GP model augmented with inducing points. The optimization is done on those points, allowing for stochastic updates and huge scalability. The counterpart can be a slightly lower accuracy and the need to select the number and the location of the inducing points (however this is a problem currently worked on).

### Regression

For **regression** one can use the `Gaussian` or `StudentT` likelihood. The first one is the vanilla GP with **Gaussian noise** while the second is using the __Student-T__ likelihood and is therefore a lot more robust to ouliers.

### Classification

For **classification** one can select a Bernoulli likelihood with a [**logistic link**](https://en.wikipedia.org/wiki/Logistic_function) or the `BayesianSVM` model based on the [**frequentist SVM**](https://en.wikipedia.org/wiki/Support_vector_machine#Bayesian_SVM).

### Model creation

Creating a model is as simple as doing `VGP(X,y,kernel,likelihood,inference;args...)` where `args` is described in the API. The compatibility of likelihoods and inferences is described in the next section and is regularly updated. For the kernels check out the kernel section

### Compatibility table

| Likelihood/Inference | AnalyticInference | GibbsSampling | NumericalInference(Quadrature) | NumericalInference(MCMCIntegration) |
| --- | :-: | :-: | :-: | :-: |
| GaussianLikelihood   | ✔  | ✖  | ✖ | ✖  |
| StudentTLikelihood   | ✔  | (dev)  | (dev) | ✖  |
| LogisticLikelihood   | ✔  | ✔  | (dev) | ✖  |
| BayesianSVM   | ✔  | ✖  | ✖ | ✖  |
| LogisticSoftMaxLikelihood   | ✔  | ✔  | ✖ | (dev)  |
| SoftMaxLikelihood   | ✖  |  ✖  | ✖  | (dev)  |


## [Training](@id train)

Training is straightforward after initializing the model by running :
```julia
train!(model;iterations=100,callback=callbackfunction)
```
Where the `callback` option is for running a function at every iteration. `callback function should be defined as`
```julia
function callbackfunction(model,iter)
    "do things here"...
end
```

## [Prediction](@id pred)

Once the model has been trained it is finally possible to compute predictions. There always three possibilities :

- `predict_f(model,X_test,covf=true,fullcov=false)` : Compute the parameters (mean and covariance) of the latent normal distributions of each test points. If `covf=false` return only the mean, if `fullcov=true` return a covariance matrix instead of only the diagonal
- `predict_y(model,X_test)` : Compute the point estimate of the predictive likelihood for regression or the label of the most likely class for classification.
- `proba_y(model,X_test)` : Return the mean with the variance of eahc point for regression or the predictive likelihood to obtain the class `y=1` for classification.

## Miscellaneous

In construction
<!-- ### Saving/Loading models

Once a model has been trained it is possible to save its state in a file by using  `save_trained_model(filename,model)`, a partial version of the file will be save in `filename`.

It is then possible to reload this file by using `load_trained_model(filename)`. **!!!However note that it will not be possible to train the model further!!!** This function is only meant to do further predictions.

### Pre-made callback functions

There is one (for now) premade function to return a a MVHistory object and callback function for the training of binary classification problems.
The callback will store the ELBO and the variational parameters at every iterations included in iter_points
If X_test and y_test are provided it will also store the test accuracy and the mean and median test loglikelihood -->
