using AugmentedGaussianProcesses
using Plots
using Distributions, LinearAlgebra

X = range(0,10, length=100)
k = SqExponentialKernel()
K = kernelmatrix(k,X) + 1e-5I
fs = [rand(MvNormal(K)) for _ in 1:2]

plot(X,fs,lab="",lw=3.0)

p = 3
xs = [hcat([f[i:end-p+i] for i in 1:p]...) for f in fs]
_fs = [f[1:end-p+1] for f in fs]
m = AGP.MOVGP(xs,_fs,SqExponentialKernel(),GaussianLikelihood(),AnalyticVI(),2)
train!(m, 10)
