using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using KernelFunctions
using MLDataUtils
using Distributions, LinearAlgebra
using Random: seed!
# seed!(33)
N = 100
x = collect(range(-3,3,length=N))
X = reshape(x,:,1)
_,y1 = noisy_function(sinc,x,noise=0.01)
_,y2 = noisy_function(sech,x)

k = SqExponentialKernel()
m = AGP.MOSVGP(X,[y1,y2],k,GaussianLikelihood(0.1,opt_noise=false),AnalyticVI(),3,10,optimizer=true,verbose=3)
# for gp in m.f
#     gp.μ .= randn(model.nFeatures)
# end
cb(model,iter) = display(ELBO(model))
# cb(model,iter) = display(heatmap(model.A[:,1,:],yflip=true))
# m.A = zeros(2,1,m.nLatent)
# m.A[1,1,1] = 1.0
# m.A[2,1,2] = 0.5
# m.A[2,1,3] = 0.0
train!(m,1000,callback=cb)
mm1 = SVGP(X,y1,k,GaussianLikelihood(),AnalyticVI(),10)
train!(mm1,100,callback=nothing)
mm2 = SVGP(X,y2,k,GaussianLikelihood(),AnalyticVI(),10)
train!(mm2,100,callback=nothing)
##
(f_1,sig_f1),(f_2,sig_f2) = predict_f(m,X,covf=true)
y_1,y_2 = predict_y(m,X)
(y_1,sig_1),(y_2,sig_2) = proba_y(m,X)

fs_1 = predict_f(mm1,X,covf=false)
ys_1 = predict_y(mm1,X)
(ys_1,sigs_1) = proba_y(mm1,X)
fs_2 = predict_f(mm2,X,covf=false)
ys_2 = predict_y(mm2,X)
(ys_2,sigs_2) = proba_y(mm2,X)

using Plots; gr()
p = scatter(x,y1,lab="true y_1",color=1)
scatter!(x,y2,lab="true y_2",color=2)
plot!(x,y_1,lab="pred y_1",lw=3.0,color=1)
plot!(x,ys_1,lab="pred ys_1",lw=3.0,color=1,linestyle=:dash)
plot!(x,y_2,lab="pred y_2",lw=3.0,color=2)
plot!(x,ys_2,lab="pred ys_2",lw=3.0,color=2,linestyle=:dash)
plot!(x,y_1+2*sqrt.(sig_1),fillrange=y_1-2sqrt.(sig_1),color=1,alpha=0.3,lab="")
plot!(x,y_2+2*sqrt.(sig_2),fillrange=y_2-2sqrt.(sig_2),color=2,alpha=0.3,lab="")
plot!(x,AGP.mean_f(m)[1][1])
plot!(x,AGP.mean_f(m)[2][1])
scatter!(AGP.get_X(m),collect(AGP.get_μ(m)))  |> display
# p
# plot(x,collect(AGP.get_μ(m)))
##

mclass = VGP(X,rand(1:3,size(X,1)),k,NegBinomialLikelihood(),AnalyticVI())
mclass.y

train!(mclass)
typeof(mclass)
AGP.predict_f(mclass,X,covf=true)

mclass.X
