# User Guide

There are 3 main actions needed to train and use the different models:

- [Initialization](@ref init)
- [Training](@ref train)
- [Prediction](@ref pred)

## [Initialization](@id init)

### Possible models

There are currently 8 possible Gaussian Process models:
#### [`GP`](@ref)
GP corresponds to the original GP regression model, it is necessarily with a Gaussian likelihood.
```julia
    GP(X_train, y_train, kernel; kwargs...)
```
#### [`VGP`](@ref)
VGP is a variational GP model: a multivariate Gaussian is approximating the true posterior. There is no inducing points augmentation involved. Therefore it is well suited for small datasets (~10^3 samples).
```julia
    VGP(X_train, y_train, kernel, likelihood, inference; kwargs...)
```
#### [`SVGP`](@ref)
SVGP is a variational GP model augmented with inducing points. The optimization is done on those points, allowing for stochastic updates and large scalability. The counterpart can be a slightly lower accuracy and the need to select the number and the location of the inducing points (however this is a problem currently worked on).
```julia
    SVGP(kernel, likelihood, inference, Z; kwargs...)
```
Where `Z` is the position of the inducing points.

#### [`MCGP`](@ref)
MCGP is a GP model where the posterior is represented via a collection of samples.
```julia
   MCGP(X_train, y_train, kernel, likelihood, inference; kwargs...)
```

#### [`OnlineSVGP`](@ref)
OnlineSVGP is an online variational GP model. It is based on the streaming method of Bui 17', it supports all likelihoods, even with multiple latent GPs.
```julia
    OnlineSVGP(kernel, likelihood, inference, ind_point_algorithm; kwargs...)
```

#### [`MOVGP`](@ref)
MOVGP is a multi output variational GP model based on the principle `f_output[i] = sum(A[i, j] * f_latent[j] for j in 1:n_latent)`. The number of latent GP is free.
```julia
    MOVGP(X_train, ys_train, kernel, likelihood/s, inference, n_latent; kwargs...)
```

#### [`MOSVGP`](@ref)
MOSVGP is the same thing as `MOVGP` but with inducing points: a multi output sparse variational GP model, based on Moreno-Muñoz 18'.
```julia
    MOVGP(kernel, likelihood/s, inference, n_latent, n_inducing_points; kwargs...)
```

#### [`VStP`](@ref)
VStP is a variational Student-T model where the prior is a multivariate Student-T distribution with scale `K`, mean `μ₀` and degrees of freedom `ν`.
The inference is done automatically by augmenting the prior as a scale mixture of inverse gamma.
```julia
    VStP(X_train, y_train, kernel, likelihood, inference, ν; kwargs...)
```
### [Likelihood](@id likelihood_user)

`GP` can only have a Gaussian likelihood, while the other have more choices. Here are the ones currently implemented:

#### Regression

For **regression**, four likelihoods are available :
- The classical [`GaussianLikelihood`](@ref), for [**Gaussian noise**](https://en.wikipedia.org/wiki/Gaussian_noise).
- The [`StudentTLikelihood`](@ref), assuming noise from a [**Student-T**](https://en.wikipedia.org/wiki/Student%27s_t-distribution) distribution (more robust to ouliers).
- The [`LaplaceLikelihood`](@ref), with noise from a [**Laplace**](https://en.wikipedia.org/wiki/Laplace_distribution) distribution.
- The [`HeteroscedasticLikelihood`](@ref), (in development) where the noise is a function of the input: ``Var(X) = λσ^{-1}(g(X))`` where `g(X)` is an additional Gaussian Process and `σ` is the logistic function.

#### Classification

For **classification** one can select among
- The [`LogisticLikelihood`](@ref) : a Bernoulli likelihood with a [**logistic link**](https://en.wikipedia.org/wiki/Logistic_function).
- The [`BayesianSVM`](@ref) likelihood based on the [**frequentist SVM**](https://en.wikipedia.org/wiki/Support_vector_machine#Bayesian_SVM), equivalent to use a hinge loss.

#### Event Likelihoods

For likelihoods such as Poisson or Negative Binomial, we approximate a parameter by `σ(f)`. Two Likelihoods are implemented :
- The [`PoissonLikelihood`](@ref) : A discrete [Poisson process](https://en.wikipedia.org/wiki/Poisson_distribution) (one parameter per point) with the scale parameter defined as `λσ(f)`.
- The [`NegBinomialLikelihood`](@ref) : The [Negative Binomial likelihood](https://en.wikipedia.org/wiki/Negative_binomial_distribution) where `r` is fixed and we define the success probability `p` as `σ(f)`.

#### Multi-class classification

There is two available likelihoods for multi-class classification:
- The [`SoftMaxLikelihood`](@ref), the most common approach. However no analytical solving is possible.
- The [`LogisticSoftMaxLikelihood`](@ref), a modified softmax where the exponential function is replaced by the logistic function. It allows to get a fully conjugate model, [**Corresponding paper**](https://arxiv.org/abs/1905.09670).

### More options

There is the project to get distributions from `Distributions.jl` to work directly as likelihoods.

### Inference

Inference can be done in various ways.

- [`AnalyticVI`](@ref) : [Variational Inference](https://en.wikipedia.org/wiki/Variational_Bayesian_methods) with closed-form updates. For non-Gaussian likelihoods, this relies on augmented version of the likelihoods. For using Stochastic Variational Inference, one can use [`AnalyticSVI`](@ref) with the size of the mini-batch as an argument.
- [`GibbsSampling`](@ref) : Gibbs Sampling of the true posterior, this also rely on an augmented version of the likelihoods, this is only valid for the `VGP` model at the moment.

The two next methods rely on numerical approximation of an integral and I therefore recommend using the classical `Descent` approach as it will use anyway the natural gradient updates. `ADAM` seem to give random results.
- [`QuadratureVI`](@ref) : Variational Inference with gradients computed by estimating the expected log-likelihood via quadrature.
- [`MCIntegrationVI`](@ref) : Variational Inference with gradients computed by estimating the expected log-likelihood via Monte Carlo Integration.

[WIP] : [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl) will be integrated at some point, although generally the Gibbs sampling is preferable when available.

### [Compatibility table](@id compat_table)

Not all inference are implemented/valid for all likelihoods, here is the compatibility table between them.

| Likelihood/Inference | AnalyticVI | GibbsSampling | QuadratureVI | MCIntegrationVI |
| --- | :-: | :-: | :-: | :-: |
| GaussianLikelihood   | ✔ (Analytic)  | ✖  | ✖ | ✖  |
| StudentTLikelihood   | ✔  | ✔ | ✔ | ✖  |
| LaplaceLikelihood   | ✔ | ✔ | ✔ | ✖ |
| HeteroscedasticLikelihood   | ✔ | ✔  | (dev)  | ✖ |
| LogisticLikelihood   | ✔  | ✔  | ✔ | ✖  |
| BayesianSVM   | ✔  | (dev) | ✖ | ✖  |
| LogisticSoftMaxLikelihood   | ✔  | ✔  | ✖ | (dev)  |
| SoftMaxLikelihood   | ✖  |  ✖  | ✖  | ✔  |
| Poisson   | ✔ | ✔ | ✖  |  ✖ |
| NegBinomialLikelihood   | ✔ | ✔ | ✖  |  ✖ |
|   |   |   |   |   |
(dev) means that the feature is possible and may be developped and tested but is not available yet. All contributions or requests are very welcome!

| Model/Inference | AnalyticVI | GibbsSampling | QuadratureVI | MCIntegrationVI |
| --- | :-: | :-: | :-: | :-: |
| VGP | ✔ | ✖ | ✔ | ✔ |
| SVGP | ✔ | ✖ | ✔ | ✔ |
| MCGP | ✖ | ✔ | ✖ | ✖ |
| OnlineSVGP | ✔ | ✖ | ✖ | ✖ |
| MO(S)VGP | ✔ | ✖ | ✔ | ✔ |
| VStP | ✔ | ✖ | ✔ | ✔ |

Note that for MO(S)VGP you can use a mix of different likelihoods.
### Inducing Points

Both [`SVGP`](@ref) and [`MOSVGP`](@ref) do not take data directly as inputs but inducing points instead.
AGP.jl directly reexports the [InducingPoints.jl](https://github.com/JuliaGaussianProcesses/InducingPoints.jl) package for you to use.
For example to use a k-means approach to select `100` points on your input data you can use:
```julia
    Z = inducingpoints(KmeanAlg(100), X)
```

`Z` will always be an `AbstractVector` and be directly compatible with `SVGP` and `MOSVGP`

For [`OnlineSVGP`](@ref), since it cannot be assumed that you have data from the start, only an [online inducing points selection algorithm](https://juliagaussianprocesses.github.io/InducingPoints.jl/dev/#Online-Inducing-Points-Selection) can be used.
The inducing points locations will be initialized with the first batch of data
### Additional Parameters

#### Hyperparameter optimization

One can optimize the kernel hyperparameters as well as the inducing points location by maximizing the ELBO. All derivations are already hand-coded (no AD needed). One can select the optimization scheme via :
- The `optimiser` keyword, can be `nothing` or `false` for no optimization or can be an optimiser from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/).
- The `Zoptimiser` keyword, similar to `optimiser` it is used for optimizing the inducing points locations, it is by default set to `nothing` (no optimization).

#### [PriorMean](@id meanprior)

The `mean` keyword allows you to add different types of prior means:
- [`ZeroMean`](@ref), a constant mean that cannot be optimized.
- [`ConstantMean`](@ref), a constant mean that can be optimized.
- [`EmpiricalMean`](@ref), a vector mean with a different value for each point.
- [`AffineMean`](@ref), `μ₀` is given by `X*w + b`.

## [Training](@id train)


### Offline models

Training is straightforward after initializing the `model` by running :
```julia
model, state = train!(model, X_train, y_train; iterations=100, callback=callbackfunction)
```
where the `callback` option is for running a function at every iteration. `callbackfunction` should be defined as
```julia
function callbackfunction(model, iter)
    # do things here...
end
```

The returned `state` will contain different variables such as some kernel matrices and local variables.
You can reuse this state to save some computations when using prediction functions or computing the [`ELBO`](@ref).

Note that passing `X_train` and `y_train` is optional for [`GP`](@ref), [`VGP`](@ref) and [`MCGP`](@ref)

### Online models

We recommend looking at [the tutorial on online Gaussian processes](/examples/onlinegp/).
One needs to pass a state around, i.e.
```julia
    let state=nothing
        for (X_batch, y_batch) in eachbatch((X_train, y_train))
            model, state = train!(model, X_batch, y_batch, state; iterations=10)
        end
    end
```

## [Prediction](@id pred)

Once the model has been trained it is finally possible to compute predictions. There always three possibilities :

- `predict_f(model, X_test; covf=true, fullcov=false)` : Compute the parameters (mean and covariance) of the latent normal distributions of each test points. If `covf=false` return only the mean, if `fullcov=true` return a covariance matrix instead of only the diagonal.
- `predict_y(model, X_test)` : Compute the point estimate of the predictive likelihood for regression or the label of the most likely class for classification.
- `proba_y(model, X_test)` : Return the mean with the variance of each point for regression or the predictive likelihood to obtain the class `y=1` for classification.

## Miscellaneous

🚧 **In construction -- Should be developed in the near future** 🚧

Saving/Loading models

Once a model has been trained it is possible to save its state in a file by using  `save_trained_model(filename,model)`, a partial version of the file will be save in `filename`.

It is then possible to reload this file by using `load_trained_model(filename)`. **!!!However note that it will not be possible to train the model further!!!** This function is only meant to do further predictions.

🚧 Pre-made callback functions 🚧

There is one (for now) premade function to return a a MVHistory object and callback function for the training of binary classification problems.
The callback will store the ELBO and the variational parameters at every iterations included in iter_points
If `X_test` and `y_test` are provided it will also store the test accuracy and the mean and median test loglikelihood
