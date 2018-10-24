"""
*Gradient Descent optimizers for Julia.*

# Introduction

This package abstracts the "boilerplate" code necessary for gradient descent. Gradient descent is "a way to minimize an objective function ``J(θ)`` parameterized by a model's parameters ``θ ∈ Rᵈ``" (Ruder 2017). Gradient descent finds ``θ`` which minizes ``J`` by iterating over the following update

``θ = θ - η ∇J(θ)``

until convergence of ``θ``. Certainly, the gradient calculation is model specific, however the learning rate ``η`` (at a given iteration) is not. Instead there are many different gradient descent variants which determine the learning rate. Each type of gradient descent optimizer has its own pros/cons. For most of these optimizers, the calculation of the learning rate is based on the value of the gradient (evaluated at a particular ``θ``) and a few (unrelated to the model) hyperparameters.

The purpose of this package is to allow the user to focus on the calculation of the gradients and not worry about the code for the gradient descent optimizer. I envision a user implementing his/her gradients, experimenting with various optimizers, and modifying the gradients as necessary.

# Examples

## Quadratic Function

Here I demonstrate a very simple example - minimizing ``x²``. In this example, I use "Adagrad", a common gradient descent optimizer.

```julia
using GradDescent

# objective function and gradient of objective function
J(x) = x ^ 2
dJ(x) = 2 * x

# number of epochs
epochs = 1000

# instantiation of Adagrad optimizer with learning rate of 1.0
# note that this learning rate is likely to high for a
# high dimensional case
opt = Adagrad(η=1.0)

# initial value for x (usually initialized with a random value)
x = 20.0

for i in 1:epochs
    # calculate the gradient wrt to the current x
    g = dJ(x)

    # change to the current x
    δ = update(opt, g)
    x -= δ
end
```

## Linear Regression

Next I demonstrate a more common example - determining the coefficients of a linear model. Here I use "Adam" an extension of "Adagrad". In this example, we minimize the mean squared error of the predicted outcome and the actual outcome. The parameter space is the coefficients of the regression model.

```julia
using GradDescent, Distributions, ReverseDiff

srand(1) # set seed
n = 1000 # number of observations
d = 10   # number of covariates
X = rand(Normal(), n, d) # simulated covariates
b = rand(Normal(), d)    # generated coefficients
ϵ = rand(Normal(0.0, 0.1), n) # noise
Y = X * b + ϵ # observed outcome
obj(Y, X, b) = mean((Y - X * b) .^ 2) # objective to minimize

epochs = 100 # number of epochs

θ = rand(Normal(), d) # initialize model parameters
opt = Adam(α=1.0)  # initalize optimizer with learning rate 1.0

for i in 1:epochs
    # here we use automatic differentiation to calculate
    # the gradient at a value
    # an analytically derived gradient is not required
    g = ReverseDiff.gradient(θ -> obj(Y, X, θ), θ)

    δ = update(opt, g)
    θ -= δ
end
```

## Variational Inference

Finally, I end with an example of black box variational inference (which is what initially motivated this package). Variational inference is a framework for approximating Bayesian posterior distributions using optimization. Most recent algorithms involve monte carlo estimation of gradients in conjuction with gradient ascent. Using `GradDescent`, we can focus on the gradient calculation without worrying too much about tracking learning rate parameters.

In this example we perform a full bayesian analysis on a simple model - normally distribution data with known variance. We place a "noninformative" Normal prior on the mean.

```julia
using Distributions, ForwardDiff, GradDescent, StatsFuns

srand(1) # set seed
n = 1000 # number of observations
μ_true = 3.0  # true mean
σ_true = 1.0 # true standard deviation
x = rand(Normal(μ_true, σ_true), n) # simulate data

# prior on mean
prior = Normal(0.0, 100.0)

# initialize variational parameters
λ = rand(Normal(), 1, 2)
λ[2] = softplus(λ[2])

# initialize optimizer
opt = Adam(α=1.0)

S = 10 # number of monte carlo simulations for gradient estimation
epochs = 50 # number of epochs

for i in 1:epochs
    # draw S samples from q
    z = rand(Normal(λ[1], softplus(λ[2])), S)

    # joint density calculations
    log_prior = logpdf(prior, z)
    log_lik = map(zi -> loglikelihood(Normal(zi, σ_true), x), z)
    joint = log_prior + log_lik

    # log variational densities
    entropy = logpdf(Normal(λ[1], softplus(λ[2])), z)

    # score function calculations
    qg = ForwardDiff.jacobian(x -> logpdf(Normal(x[1], x[2]),
                                          z),
                              [λ[1], softplus(λ[2])])

    # construct monte carlo samples st the expected value is the gradient
    # of the ELBO
    f = qg .* (joint - entropy)
    h = qg
    a = sum(diag(cov(f, h))) / sum(diag(cov(h)))
    g = mean(f - a * h, 1) # compute gradient

    # perform gradient ascent step
    δ = update(opt, g)
    λ += δ

    # truncate variational standard deviation
    # don't allow it to be too close to 0.0
    λ[2] = λ[2] < invsoftplus(1e-5) ? invsoftplus(1e-5) : λ[2]
end

# after gradient ascent, apply softplus function
λ[2] = softplus(λ[2])
```

"""
module GradDescent

using LinearAlgebra
export
    Optimizer,
    VanillaGradDescent,
    Momentum,
    Adagrad,
    Adadelta,
    RMSprop,
    Adam,
    Adamax,
    Nadam,
    update,
    t

include("AbstractOptimizer.jl")
include("VanillaGradDescent.jl")
include("MomentumOptimizer.jl")
include("AdaGradOptimizer.jl")
include("AdaDeltaOptimizer.jl")
include("RMSpropOptimizer.jl")
include("AdamOptimizer.jl")
include("AdamaxOptimizer.jl")
include("NadamOptimizer.jl")

end # module
