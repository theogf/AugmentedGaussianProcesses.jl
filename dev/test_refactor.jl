using AugmentedGaussianProcesses; const AGP= AugmentedGaussianProcesses
using MLDataUtils, LinearAlgebra, PDMats
using KernelFunctions

X,f = noisy_function(sinc,range(-3,3,length=100))
y = sign.(f)

function train_and_ELBO(vals)
    # m = SVGP(X,y,SqExponentialKernel(l),LogisticLikelihood(),AnalyticVI(),10,optimizer=false)
    AGP.expec_logpdf(QuadratureVI(),LogisticLikelihood(),vals.μ,vals.Σ,y)
end

using Zygote

struct gpstore
    μ
    Σ
end
v = gpstore(zeros(Float64,length(y)),diagm(ones(Float64,length(y))))
train_and_ELBO(v)
AGP.expec_logpdf(QuadratureVI(),LogisticLikelihood(),v.μ,v.Σ,y)
AGP.apply_quad(y[1],v.μ[1],v.Σ[1,1],QuadratureVI(),LogisticLikelihood())
Zygote.refresh()
Zygote.gradient(train_and_ELBO,v)

nodes,weights = AGP.gausshermite(100)
y = rand(100)
using Distributions
function bar(gh,m,y)
    x = gh[1] .+ m
    dot(gh[2],exp.(y.*x))
end
function foo(v)
    Zygote.@showgrad w = bar.([(nodes,weights)],v,y)
    sum(w)
end
foo(rand(100))
Zygote.refresh()
Zygote.gradient(foo,rand(100))


using Plots
function cb(model,iter)
    if iter%50 != 0
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
M = VGP(X,y,SqExponentialKernel(),LogisticLikelihood(),QuadratureVI(),optimizer=true,verbose=3)
# cb(model,iter) = @info "L = $(ELBO(model)), k_l = $(get_params(model.f[1].kernel)), σ = $(model.f[1].σ_k)"
train!(M,1000,callback=cb)
using GradDescent
m = SVGP(X,y,SqExponentialKernel(),LogisticLikelihood(),QuadratureVI(),10,optimizer=false,verbose=3,Zoptimizer=nothing)
# m.f[1].Z.opt = Adam(α=0.01)

train!(m,1000,callback=cb)
ELBO(m)

##
pred_F,sig_F = predict_f(M,X,covf=true)
pred_f = predict_f(m,X,covf=false)
pred_X = predict_y(M,X)
pred_x = predict_y(m,X)
proba_X,_ = proba_y(M,X)
proba_x,_ = proba_y(m,X)
scatter(X,y)
scatter!(X,pred_x)
scatter!(X,pred_X)
scatter!(AGP.get_X(m)[:],zeros(length(AGP.get_X(m))))
plot!(X,pred_f)
plot!(X,pred_F)
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
