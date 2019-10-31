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
m = AGP.MOSVGP(X,[y1,y2],k,GaussianLikelihood(opt_noise=false),AnalyticVI(),3,10,optimizer=false)
cb(model,iter) = display(model.A[:,1,:])
m.A = zeros(2,1,3)
m.A[1,1,1] = 1.0
m.A[2,1,2] = 1.0
train!(m,100,callback=nothing)
##
f_1,f_2 = predict_f(m,X,covf=false)
y_1,y_2 = predict_y(m,X)
(y_1,sig_1),(y_2,sig_2) = proba_y(m,X)

using Plots; gr()
p = scatter(x,y1,lab="true y_1",color=1)
scatter!(x,y2,lab="true y_2",color=2)
plot!(x,y_1,lab="pred y_1",lw=3.0,color=1)
plot!(x,y_2,lab="pred y_2",lw=3.0,color=2)
plot!(x,y_1+2*sqrt.(sig_1),fillrange=y_1-2sqrt.(sig_1),color=1,alpha=0.3,lab="")
plot!(x,y_2+2*sqrt.(sig_2),fillrange=y_2-2sqrt.(sig_2),color=2,alpha=0.3,lab="")
scatter!(AGP.get_X(m),collect(AGP.get_μ(m)))  |> display
# p
# plot(x,collect(AGP.get_μ(m)))
