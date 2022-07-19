# # Sampling from a GP
#
# ## Preliminary steps

# ### Loading necessary packages

using AugmentedGaussianProcesses
using Distributions
using LinearAlgebra
using Plots
default(lw=3.0)

# ### Generating some random binary data
kernel = with_lengthscale(SqExponentialKernel(), 1.0)
N_train = 50
x = range(-10, 10; length=50)
x_test = range(-10, 10; length=500)
K = kernelmatrix(kernel, vcat(x, x_test))
f_all = rand(MvNormal(K + 1e-8I)) # Sample a random GP
f = f_all[1:N_train]
y = rand.(Bernoulli.(AGP.logistic.(f)))
y_sign = Int.(sign.(y .- 0.5));

# ### We create a function to visualize the data

function plot_data(x, y; size=(300, 500), kwargs...)
    return Plots.scatter(x, y; alpha=0.5, markerstrokewidth=0.0, lab="", size=size, kwargs...)
end
plot_data(x, y; size=(600, 500), xlabel="x", ylabel="y")
# ## Model initialization and training
# ### Run the variational gaussian process approximation
@info "Running full model"
mfull = VGP(x, y_sign, kernel, LogisticLikelihood(), AnalyticVI(); optimiser=false)
@time train!(mfull, 5)

# ### We can also create a sampling based model
@info "Sampling from model"
mmcmc = MCGP(x, y, kernel, LogisticLikelihood(), GibbsSampling(); optimiser=false)
m = mmcmc
@time samples = sample(mmcmc, 1000);

# ### We can now visualize the results of both models

# ### We first plot the latent function f (truth, the VI estimate, the samples)
p1 = plot(x, samples; label="", color=:black, alpha=0.01, lab="")
plot!(x, mean(mfull[1]); color=:blue, ribbon=sqrt.(var(mfull[1])), label="VI")
plot!(x_test, f_all[N_train+1:end]; color=:red, label="true f")
# ### And we can also plot the predictions vs the data
p2 = plot_data(x, y; size=(600, 400))
μ_vi, σ_vi = proba_y(mfull, x_test)
plot!(x_test, μ_vi; ribbon=σ_vi, label="VI")
μ_mcmc, σ_mcmc = proba_y(mmcmc, x_test)
plot!(x_test, μ_mcmc; ribbon=σ_mcmc, label="MCMC")
