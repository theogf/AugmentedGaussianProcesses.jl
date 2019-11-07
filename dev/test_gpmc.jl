using AugmentedGaussianProcesses; const AGP= AugmentedGaussianProcesses
using MLDataUtils, LinearAlgebra, PDMats
using KernelFunctions

X,f = noisy_function(sinc,range(-3,3,length=100))
y = sign.(f)
##
using Plots; pyplot()
function cb(model,iter)
    if iter%10 != 0
        return
    end
    # pred_f = predict_f(model,X,covf=false)
    # proba_x,_ = proba_y(model,X)
    # p = scatter(X,y)
    # if isa(model,SVGP)
    #     scatter!(AGP.get_X(m)[:],zeros(length(AGP.get_X(m))))
    # end
    # plot!(X,pred_f)
    # plot!(X,proba_x)
    # display(p)
end
using GradDescent
M = MCGP(X,y,SqExponentialKernel(),LogisticLikelihood(),GibbsSampling(),optimizer=false,verbose=3,variance=10.0)
# cb(model,iter) = @info "L = $(ELBO(model)), k_l = $(get_params(model.f[1].kernel)), σ = $(model.f[1].σ_k)"
sample(M,100,callback=nothing)

##
# pred_F,sig_F = predict_f(M,X,covf=true)
# pred_f,sig_f = predict_f(m,X,covf=true)
# pred_X = predict_y(M,X)
# pred_x = predict_y(m,X)
# proba_X,_ = proba_y(M,X)
# proba_x,_ = proba_y(m,X)
# maximum(proba_x)
# scatter(X,y)
# scatter!(X,pred_x)
# scatter!(X,pred_X)
# scatter!(AGP.get_X(m)[:],zeros(length(AGP.get_X(m))))
# # plot!(X,pred_f)
# # plot!(X,pred_F)
# plot!(X,proba_X)
# plot!(X,proba_x)
# ##
#

# using ForwardDiff
# W = rand(100,3)
# reshape(ForwardDiff.jacobian(x->kernelmatrix(SqExponentialKernel(x)+0.5Matern32Kernel(x),W,obsdim=1),[0.5,0.2,0.1]),100,100,3)
# using Zygote
# Zygote.gradient(train_and_ELBO,0.5)
# Zygote.gradient(k->logdet(kernelmatrix(k,reshape(X,:,1),obsdim=1)),SqExponentialKernel(0.1))[1][:transform][:s][][:x]
# Zygote.gradient(k->logdet(kernelmatrix(k,reshape(X,:,1),obsdim=1)),SqExponentialKernel([0.1]))[1][:transform][:s]
