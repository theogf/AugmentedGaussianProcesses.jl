```@meta
EditURL = "<unknown>/docs/examples/gpregression.jl"
```

# Regression with a Gaussian Likelihood

## Use necessary packages

```@example gpregression
using AugmentedGaussianProcesses
const AGP = AugmentedGaussianProcesses
using Distributions
using Plots; pyplot()
```

We create a toy dataset with X ∈ [-20, 20] and y = 5 * sinc(X)

```@example gpregression
N = 1000
X = reshape((sort(rand(N)) .- 0.5) * 40.0, N, 1)
σ = 0.01

function latent(x)
    5.0 * sinc.(x)
end
Y = vec(latent(X) + σ * randn(N))
```

Visualization of the data :

```@example gpregression
scatter(X, Y, lab="")
```

## Gaussian noise

In this first example we are going to look at the effect of using
inducing points compared to the true Gaussian Process
For simplicity we will keep all inducing points and kernel parameters fixed

Run sparse classification with an increasing number of inducing points

```@example gpregression
Ms = [4, 8, 16, 32, 64]
```

Create an empty array of GPs

```@example gpregression
models = Vector{AbstractGPModel}(undef,length(Ms) + 1)
kernel = SqExponentialKernel()#  + PeriodicKernel()
for (index, num_inducing) in enumerate(Ms)
    @info "Training with $(num_inducing) points"
    m = SVGP(X, Y, # First arguments are the input and output
            kernel, # Kernel
            GaussianLikelihood(σ), # Likelihood used
            AnalyticVI(), # Inference usede to solve the problem
            num_inducing; # Number of inducing points used
            optimiser = false, # Keep kernel parameters fixed
            Zoptimiser = false, # Keep inducing points locations fixed
    )
    @time train!(m, 100) # Train the model for 100 iterations
    models[index] = m # Save the model in the array
end
```

Train the model without any inducing points (no approximation)

```@example gpregression
@info "Training with full model"
mfull = GP(X, Y, kernel,
            noise = σ,
            opt_noise = false, # Keep the noise value fixed
            optimiser = false, # Keep kernel parameters fixed
            )
@time train!(mfull, 5);
models[end] = mfull;
nothing #hide
```

Create a grid and compute prediction on it

```@example gpregression
function compute_grid(model, n_grid=50)
    mins = -20; maxs = 20
    x_grid = range(mins, maxs, length = n_grid) # Create a grid
    y_grid, sig_y_grid = proba_y(model, reshape(x_grid, :, 1)) # Predict the mean and variance on the grid
    return y_grid, sig_y_grid, x_grid
end
```

Plot the data as a scatter plot

```@example gpregression
function plotdata(X,Y)
    return Plots.scatter(X, Y, alpha=0.33,
                msw=0.0, lab="", size=(300,500))
end

function plot_model(model, X, Y, title = nothing)
    n_grid = 100
    y_grid, sig_y_grid, x_grid = compute_grid(model,n_grid)
    title = if isnothing(title)
        (model isa SVGP ? "M = $(dim(model[1]))" : "full")
    else
        title
    end

    p = plotdata(X, Y)
    Plots.plot!(p, x_grid, y_grid,
                ribbon=2 * sqrt.(sig_y_grid), # Plot 2 std deviations
                title=title,
                color="red",
                lab="",
                linewidth=3.0)
    if model isa SVGP # Plot the inducing points as well
        Plots.plot!(p,
                    vec(model.f[1].Z),
                    zeros(dim(model.f[1])),
                    msize=2.0,
                    color="black",t=:scatter,lab="")
    end
    return p
end;

Plots.plot(plot_model.(models, Ref(X), Ref(Y))...,
            layout=(1, length(models)),
            size=(1000,200)
        ) # Plot all models and combine the plots
```

## Non-Gaussian Likelihoods
We now look at using another noise than Gaussian noise.
In AGP.jl you can use the Student-T likelihood,
the Laplace likelihood and the Heteroscedastic likelihood

We will use the same toy dataset for our experiment

Create an array of model with different likelihoods:

```@example gpregression
likelihoods = [StudentTLikelihood(3.0),
                LaplaceLikelihood(3.0),
                HeteroscedasticLikelihood(1.0)]
ngmodels = Vector{AbstractGPModel}(undef, length(likelihoods)+1)
for (i, l) in enumerate(likelihoods)
    @info "Training with the $(l)" # We need to use VGP
    m = VGP(X, Y, # First arguments are the input and output
            kernel, # Kernel
            l, # Likelihood used
            AnalyticVI(), # Inference usede to solve the problem
            optimiser = false, # Keep kernel parameters fixed
    )
    @time train!(m, 10) # Train the model for 100 iterations
    ngmodels[i] = m # Save the model in the array
end

ngmodels[end] = models[end] # Add the Gaussian model
```

We can now repeat the prediction from before :

```@example gpregression
Plots.plot(plot_model.(ngmodels, Ref(X), Ref(Y), ["Student-T", "Laplace", "Heteroscedastic", "Gaussian"])...,
            layout=(1, length(ngmodels)),
            size=(1000,200)
        ) # Plot all models and combine the plots
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

