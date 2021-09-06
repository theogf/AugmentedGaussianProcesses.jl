using AugmentedGaussianProcesses
using Distributions, LinearAlgebra
using Random: seed!;
seed!(123);
using Plots;
gr();

N = 100;
noise = 1e-1;
N_test = 1000
X = sort(rand(N))
X_test = range(0, 1; length=N_test)
k = SqExponentialKernel() ∘ ScaleTransform(10.0)
K = kernelmatrix(k, X) + noise * I
f = rand(MvNormal(K))
y = sign.(f)

logitmodel = VGP(X, y, k, LogisticLikelihood(), AnalyticVI());
train!(logitmodel, 300)
logitpred = (proba_y(logitmodel, X_test)[1] .- 0.5) .* 2

svmmodel = VGP(X, y, k, BayesianSVM(), AnalyticVI());
train!(svmmodel, 50)#,callback=intplot)
svmpred = (proba_y(svmmodel, X_test)[1] .- 0.5) .* 2

##
default(; legendfontsize=14.0, xtickfontsize=10.0, ytickfontsize=10.0)
p = plot(X, y; t=:scatter, lab="Training Points", ylims=(-1.5, 3.0), legend=:topright)
plot!(X_test, logitpred; lab="Logistic Prediction", lw=7.0)
plot!(X_test, svmpred; lab="BayesianSVM Prediction", lw=7.0)

display(p)
savefig(p, joinpath(@__DIR__, "Classification.png"))
