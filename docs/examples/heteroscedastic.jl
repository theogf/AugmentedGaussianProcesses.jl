# # Gaussian Process with Heteroscedastic likelihoods
# 
# ### Loading necessary packages
using AugmentedGaussianProcesses
using Distributions
# using CairoMakie

# ## Model generated data
# The heteroscedastic noise mean that the variance of the likelihood
# directly depends on the input.
# To model this with Gaussian process, we define a GP `f` for the mean and another GP `g` for the variance
# ``y \sim f + \epsilon``
# where ``\epsilon \sim \mathcal{N}(0, (\lambda \sigma(g))^{-1})``
# We create a toy dataset with X ∈ [-20, 20] and sample `f`, `g` and `y` given this same generative model
N = 200
x = (sort(rand(N)) .- 0.5) * 40.0
kernel = 5.0 * SqExponentialKernel() ∘ ScaleTransform(1.0) # Kernel function
K = kernelmatrix(kernel, x) + 1e-5I # The kernel matrix
f = rand(MvNormal(K)) # We draw a random sample from the GP prior

# We add a prior mean on `g` so that the variance does not become too big
μ₀ = -3.0
g = rand(MvNormal(μ₀ * ones(N), K))
λ = 3.0 # The maximum possible precision
σ = inv.(sqrt.(λ * AGP.logistic.(g))) # We use the following transform to obtain the std. deviation
y = f + σ .* randn(N) # We finally sample the ouput
# We can visualize the data:
n_sig = 2 # Number of std. dev. around the mean
plot(x, f; ribbon=n_sig * σ, lab="p(y|f,g)") # Mean and std. dev. of y
scatter!(x, y; alpha=0.2, msw=0.0, lab="y") # Observation samples

# ## Model creation and training
# We will now use the augmented model to infer both `f` and `g`
model = VGP(
    x,
    y,
    deepcopy(kernel),
    HeteroscedasticLikelihood(λ),
    AnalyticVI();
    optimiser=true, # We optimise both the mean parameters and kernel hyperparameters
    mean=μ₀,
    verbose=1,
)

# Model training, we train for around 100 iterations to wait for the convergence of the hyperparameters
train!(model, 100);

# ## Predictions 
# We can now look at the predictions and compare them with out original model
(f_m, g_m), (f_σ, g_σ) = predict_f(model, x; cov=true)
y_m, y_σ = proba_y(model, x)
# Let's first look at the differece between the latent `f` and `g`
plot(x, [f, g]; label=["f" "g"])
plot!(x, [f_m, g_m]; ribbon=[n_sig * f_σ, n_sig * g_σ], label=["f_pred" "g_pred"])
# But it's more interesting to compare the predictive probability of `y` directly:
plot(x, f; ribbon=n_sig * σ, lab="p(y|f,g)")
plot!(x, y_m; ribbon=n_sig * sqrt.(y_σ), lab="p(y|f,g) pred")
scatter!(x, y; lab="y", msw=0.0, alpha=0.2)
# Or to explore the heteroscedasticity itself, we can look at the residuals
scatter(x, (f - y) .^ 2; yaxis=:log, lab="residuals", msw=0.0, alpha=0.2)
plot!(x, σ .^ 2; lab="true σ²(x)", lw=3.0)
plot!(x, y_σ; lab="predicted σ²(x)", lw=3.0)
