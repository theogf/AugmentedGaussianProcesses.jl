using AugmentedGaussianProcesses;
using Distributions, LinearAlgebra
using Random: seed!;
seed!(123);
using Plots;
gr();

N = 100;
noise = 1e-1;
N_test = 1000
X = sort(rand(N))
X_test = collect(range(0; length=N_test, stop=1))
k = SqExponentialKernel() ∘ ScaleTransform(10.0)
K = kernelmatrix(k, X) + noise * I
f = rand(MvNormal(K))
λ = 10
y = rand.(Poisson.(λ * AGP.logistic.(f)))

poissonmodel = VGP(X, y, k, PoissonLikelihood(Float64(λ)), AnalyticVI(); optimiser=false);
train!(poissonmodel, 300)
poissonpred = predict_y(poissonmodel, X_test)

negbinmodel = VGP(X, y, k, NegBinomialLikelihood(λ), AnalyticVI(); optimiser=false);
train!(negbinmodel, 50)
negbinpred = predict_y(negbinmodel, X_test)
##
default(; legendfontsize=14.0, xtickfontsize=10.0, ytickfontsize=10.0)
p = plot(X, y; t=:scatter, lab="Training Points")
plot!(X_test, poissonpred; lab="Poisson Expectation", lw=7.0)
plot!(X_test, negbinpred; lab="Negative Binomial Expectation", lw=7.0, legend=:topright)

display(p)
savefig(p, joinpath(@__DIR__, "Events.png"))
# a,b = logitmodel.fstar(X_test)
# using Expectations, QuadGK, BenchmarkTools

# @btime begin; for i in 1:N_test; d = Normal(a[i],sqrt(max(b[i],1e-8))); expectation(logit,d); end; end;
# @btime begin; for i in 1:N_test; d = Normal(a[i],sqrt(max(b[i],1e-8))); quadgk(x->pdf(d,x)*logit(x),-Inf,Inf); end; end;
