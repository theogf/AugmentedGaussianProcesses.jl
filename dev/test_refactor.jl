using AugmentedGaussianProcesses; const AGP= AugmentedGaussianProcesses
using MLDataUtils, LinearAlgebra, PDMats
using KernelFunctions

X,f = noisy_function(sinc,range(-3,3,length=100),noise=0.3)
w = randn()
y = X*w + f

##
using Plots
function cb(model,iter)
    if iter%10 != 0
        return
    end
    pred_f = predict_f(model,X,covf=false)
    proba_x,_ = proba_y(model,X)
    p = scatter(X,y)
    if isa(model,SVGP)
        scatter!(AGP.get_X(m)[:],zeros(length(AGP.get_X(m))))
    end
    plot!(X,pred_f)
    plot!(X,proba_x)
    display(p)
end
using GradDescent
M = VGP(X,y,SqExponentialKernel(),LaplaceLikelihood(),AnalyticVI(),optimizer=true,verbose=3,variance=100.0,mean=AGP.AffineMean(1))
# cb(model,iter) = @info "L = $(ELBO(model)), k_l = $(get_params(model.f[1].kernel)), σ = $(model.f[1].σ_k)"
train!(M,1000,callback=nothing)
m = SVGP(X,y,SqExponentialKernel(),StudentTLikelihood(4.0),AnalyticVI(),10,optimizer=true,verbose=3,Zoptimizer=true,variance=100.0)
# m.f[1].Z.opt = Adam(α=0.01)
show_eta(model,iter) =display(heatmap(Matrix(model.f[1].η₂),yflip=true))
train!(m,100,callback=nothing)
ELBO(m)

##
pred_F,sig_F = predict_f(M,X,covf=true)
pred_f,sig_f = predict_f(m,X,covf=true)
ff = AGP._predict_f(M,reshape(X,:,1),covf=false)[1]
first(ff)
pred_X = predict_y(M,X)
pred_x = predict_y(m,X)
proba_X,sig_X = proba_y(M,X)
proba_x,sig_x = proba_y(m,X)
maximum(proba_x)
scatter(X,y,lab="data")
plot!(X,pred_x,lab="VGP",lw=3.0)
plot!(X,X*w,lab="True Linear",lw=3.0)
plot!(X,M.f[1].μ₀(reshape(X,:,1)),lab="Inferred Linear",lw=3.0)
scatter!(X,pred_X)
plot!(X,proba_X.+2*sqrt.(sig_X),fillrange=proba_X.-2*sqrt.(sig_X),alpha=0.3)
scatter!(AGP.get_Z(m)[1],zeros(length(AGP.get_Z(m)[1])))
# plot!(X,pred_f)
# plot!(X,pred_F)
plot!(X,proba_X)
plot!(X,proba_x)
##


# using ForwardDiff
# W = rand(100,3)
# reshape(ForwardDiff.jacobian(x->kernelmatrix(SqExponentialKernel(x)+0.5Matern32Kernel(x),W,obsdim=1),[0.5,0.2,0.1]),100,100,3)
# using Zygote
# Zygote.gradient(train_and_ELBO,0.5)
# Zygote.gradient(k->logdet(kernelmatrix(k,reshape(X,:,1),obsdim=1)),SqExponentialKernel(0.1))[1][:transform][:s][][:x]
# Zygote.gradient(k->logdet(kernelmatrix(k,reshape(X,:,1),obsdim=1)),SqExponentialKernel([0.1]))[1][:transform][:s]
