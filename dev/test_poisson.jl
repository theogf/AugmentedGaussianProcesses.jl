using AugmentedGaussianProcesses
using Plots, LinearAlgebra, Distributions
using SpecialFunctions
N = 1000
λ = 5.0
X = rand(N,2)
K = kernelmatrix(X,RBFKernel(0.1))
y = rand(MvNormal(zeros(N),K+1e-2I))
scatter(eachcol(X)...,zcolor=y)
n = rand.(Poisson.(λ.*AugmentedGaussianProcesses.logistic.(y)))

model = VGP(X,n,RBFKernel(0.1),PoissonLikelihood(),AnalyticVI())
train!(model,iterations=10)
predict_f(model,X)
