using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using KernelFunctions
using MLDataUtils
using Distributions, LinearAlgebra

N = 100
x = collect(range(-3,3,length=N))
X = reshape(x,:,1)
_,y1 = noisy_function(sinc,x)
_,y2 = noisy_function(sech,x)


m = MOSVGP(X,[y1,y2],GaussianLikelihood(),AnalyticVI(),3,10,optimizer=false)
train!(m,10)

using Plots; pyplot()
scatter(x,y1,lab="")
scatter!(x,y2,lab="")
