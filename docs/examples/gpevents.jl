# # Gaussian Process for Event likelihoods
# 
# ## Preliminary steps
#
# ### Loading necessary packages
using Plots
using AugmentedGaussianProcesses
const AGP = AugmentedGaussianProcesses
using Distributions

# ## Creating some random data
n_data = 200
X = (rand(n_data) .- 0.5) * 40
r = 5.0
Y = rand.(NegativeBinomial.(r, AGP.logistic.(sin.(X))))
scatter(X, Y)

# ## Run GP model with negative binomial likelihood to learn p 
kernel = transform(SqExponentialKernel(), 1.0)
m_negbinomial = VGP(
    X, Y, kernel, NegBinomialLikelihood(r), AnalyticVI(); optimiser=false, verbose=2
)
@time train!(m_negbinomial, 20)

# ## Running the same model but with a Poisson likelihood
kernel = transform(SqExponentialKernel(), 1.0)
m_poisson = VGP(
    X, Y, kernel, PoissonLikelihood(r), AnalyticVI(); optimiser=false, verbose=2
)
@time train!(m_poisson, 20)

# Prediction and plot function on a grid
# Create a grid and compute prediction on it
function compute_grid(model, n_grid=50)
    mins = -20
    maxs = 20
    x_grid = range(mins, maxs; length=n_grid) # Create a grid
    y_grid, sig_y_grid = proba_y(model, reshape(x_grid, :, 1)) # Predict the mean and variance on the grid
    return y_grid, sig_y_grid, x_grid
end

# Plot the data as a scatter plot
function plot_data(X, Y)
    return Plots.scatter(X, Y; alpha=0.33, msw=0.0, lab="", size=(800, 500))
end

function plot_model(model, X, Y, title=nothing)
    n_grid = 100
    y_grid, sig_y_grid, x_grid = compute_grid(model, n_grid)
    p = plot_data(X, Y)
    Plots.plot!(
        p,
        x_grid,
        y_grid;
        ribbon=2 * sqrt.(sig_y_grid), # Plot 2 std deviations
        title=title,
        color="red",
        lab="",
        linewidth=3.0,
    )
    if model isa SVGP # Plot the inducing points as well
        Plots.plot!(
            p,
            vec(model.f[1].Z),
            zeros(dim(model.f[1]));
            msize=2.0,
            color="black",
            t=:scatter,
            lab="",
        )
    end
    return p
end;

# ## Comparison between the two likelihoods
Plots.plot(
    plot_model.(
        [m_negbinomial, m_poisson], Ref(X), Ref(Y), ["Neg. Binomial", "Poisson"]
    )...;
    layout=(1, 2),
)
