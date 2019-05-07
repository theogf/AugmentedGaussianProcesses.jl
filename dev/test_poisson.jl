using AugmentedGaussianProcesses
using Plots, LinearAlgebra, Distributions
using SpecialFunctions
using LaTeXStrings
N = 1000
λ = 5.0
X = rand(N,2)
xrange = collect(range(0,1,length=100))
Xrange = make_grid(xrange,xrange)

K = kernelmatrix(vcat(X,Xrange),RBFKernel(0.1))
y = rand(MvNormal(zeros(N+size(Xrange,1)),K+1e-2I))
n = rand.(Poisson.(λ.*AugmentedGaussianProcesses.logistic.(y)))
n_train = n[1:N]
ptrue = contourf(xrange,xrange,reshape(n[N+1:end],100,100))
model = SVGP(X,n_train,RBFKernel(0.1),PoissonLikelihood(),AnalyticVI(),100)
train!(model,iterations=100)
norm(proba_y(model,X)-n_train)
pred_f = predict_y(model,Xrange)
pyplot()
ppred = contourf(xrange,xrange,reshape(pred_f,100,100),title=L"Prediction  \lambda")
pltrue = contourf(xrange,xrange,reshape(λ.*AugmentedGaussianProcesses.logistic.(y[N+1:end]),100,100),title=L"True \lambda")
plot(ppred,pltrue)
model.likelihood.λ[1]
mean(abs.(proba_y(model,X)-λ.*AugmentedGaussianProcesses.logistic.(y[1:N])))
mean(abs.(proba_y(model,Xrange)-λ.*AugmentedGaussianProcesses.logistic.(y[N+1:end])))
