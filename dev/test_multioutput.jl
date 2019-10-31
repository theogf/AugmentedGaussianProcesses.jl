using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using KernelFunctions
using MLDataUtils
using Distributions, LinearAlgebra

N = 100
x = collect(range(-3,3,length=N))
X = reshape(x,:,1)
_,y1 = noisy_function(sinc,x)
_,y2 = noisy_function(sech,x)

k = SqExponentialKernel()
m = AGP.MOSVGP(X,[y1,y2],k,GaussianLikelihood(),AnalyticVI(),3,10,optimizer=true)
train!(m,100)
f_1,f_2 = predict_f(m,X,covf=false)
y_1,y_2 = predict_y(m,X)

using Plots; pyplot()
scatter(x,y1,lab="true y_1")
scatter!(x,y2,lab="true y_2")
scatter!(x,y_1,lab="pred y_1")
scatter!(x,y_2,lab="pred y_2")
