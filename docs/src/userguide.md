# User Guide

There are 3 main actions needed to train and use the different models:

- [Initialization](@ref init)
- [Training](@ref train)
- [Prediction](@ref pred)

## [Initialization](@id init)

### GP vs VGP vs SVGP

There are currently 3 possible models:
- [`GP`](@ref) corresponds to the original GP regression model.
```julia
    GP(X_train,y_train,kernel)
```
- [`VGP`](@ref) is a variational GP model: a multivariate Gaussian is approximating the true posterior. There is no inducing points augmentation involved. Therefore it is well suited for small datasets (~10^3 samples)
```julia
    VGP(X_train,y_train,kernel,likelihood,inference)
```
- [`SVGP`](@ref) is a variational GP model augmented with inducing points. The optimization is done on those points, allowing for stochastic updates and large scalability. The counterpart can be a slightly lower accuracy and the need to select the number and the location of the inducing points (however this is a problem currently worked on).
```julia
    SVGP(X_train,y_train,kernel,likelihood,inference,n_inducingpoints)
```
### [Likelihood](@id likelihood_user)

`GP` can only have a Gaussian likelihood, `VGP` and `SVGP` have more choices. Here are the ones currently implemented:

#### Regression

For **regression** one can use the [`GaussianLikelihood`](@ref), the [`StudentTLikelihood`](@ref) or the [`LaplaceLikelihood`](@ref) likelihood. The first one is assuming that the model has [**Gaussian noise**](https://en.wikipedia.org/wiki/Gaussian_noise), the second assumes noise from a [**Student-T**](https://en.wikipedia.org/wiki/Student%27s_t-distribution) distribution (more robust to ouliers) and the last one assumes noise from a [**Laplace**](https://en.wikipedia.org/wiki/Laplace_distribution) distribution.

#### Classification

For **classification** one can select the [`LogisticLikelihood`](@ref) : a Bernoulli likelihood with a [**logistic link**](https://en.wikipedia.org/wiki/Logistic_function) or the [`BayesianSVM`](@ref) likelihood based on the [**frequentist SVM**](https://en.wikipedia.org/wiki/Support_vector_machine#Bayesian_SVM), equivalent to use a hinge loss.

#### Multi-class classification

In development

### Inference

Inference can be done in various ways.

- [`AnalyticVI`](@ref) : Variational Inference with closed-form updates. For non-Gaussian likelihoods, this relies on augmented version of the likelihoods. For using Stochastic Variational Inference, one can use [`AnalyticSVI`](@ref) with the size of the mini-batch as an argument
- [`GibbsSampling`](@ref) : Gibbs Sampling of the true posterior, this also rely on an augmented version of the likelihoods, this is only valid for the `VGP` model at the moment.
- [`QuadratureVI`](@ref) : Variational Inference with gradients computed by estimating the expected log-likelihood via quadrature.
- [`MCMCIntegrationVI`](@ref) : Variational Inference with gradients computed by estimating the expected log-likelihood via MCMCIntegration

### [Compatibility table](@id compat_table)

Not all inference are implemented/valid for all likelihoods, here is the compatibility table between them.

| Likelihood/Inference | AnalyticVI | GibbsSampling | QuadratureVI | MCMCIntegrationVI |
| --- | :-: | :-: | :-: | :-: |
| GaussianLikelihood   | ✔  | ✖  | ✖ | ✖  |
| StudentTLikelihood   | ✔  | ✔ | (dev) | ✖  |
| LaplaceLikelihood   | ✔ | (dev) | (dev) | ✖ |
| LogisticLikelihood   | ✔  | ✔  | (dev) | ✖  |
| BayesianSVM   | ✔  | (dev) | ✖ | ✖  |
| LogisticSoftMaxLikelihood   | ✔  | ✔  | ✖ | (dev)  |
| SoftMaxLikelihood   | ✖  |  ✖  | ✖  | (dev)  |
| Poisson   | ✔ | (dev) | ✖  |  ✖ |

(dev) means that the feature is possible and may be developped and tested but is not available yet, all contributions are very welcome!

## [Training](@id train)

Training is straightforward after initializing the `model` by running :
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

**In construction -- Should be developed in the near future**

Saving/Loading models

Once a model has been trained it is possible to save its state in a file by using  `save_trained_model(filename,model)`, a partial version of the file will be save in `filename`.

It is then possible to reload this file by using `load_trained_model(filename)`. **!!!However note that it will not be possible to train the model further!!!** This function is only meant to do further predictions.

Pre-made callback functions

There is one (for now) premade function to return a a MVHistory object and callback function for the training of binary classification problems.
The callback will store the ELBO and the variational parameters at every iterations included in iter_points
If X_test and y_test are provided it will also store the test accuracy and the mean and median test loglikelihood
