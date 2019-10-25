using AugmentedGaussianProcesses; const AGP= AugmentedGaussianProcesses
using MLDataUtils, LinearAlgebra, PDMats
using KernelFunctions

X,f = noisy_function(sinc,range(-3,3,length=100))
y = sign.(f)

M = VGP(X,y,SqExponentialKernel(),LogisticLikelihood(),AnalyticVI(),optimizer=false)
train!(M)

m = SVGP(X,y,SqExponentialKernel(),LogisticLikelihood(),AnalyticVI(),10,optimizer=false,verbose=3)
train!(m,100)
ELBO(m)


pred_F = predict_f(M,X,covf=false)
pred_f = predict_f(m,reshape(X,:,1),covf=false)
pred_X = predict_y(M,X)
pred_x = predict_y(m,X)
# using Plots
# scatter(x,y)
# scatter!(x,pred_x)
# scatter!(x,pred_X)
# scatter!(AGP.get_X(m)[:],zeros(length(AGP.get_X(m))))
# plot!(x,pred_f)
# plot!(x,pred_F)

function train_and_ELBO(l)
    m = SVGP(X,y,SqExponentialKernel(l),LogisticLikelihood(),AnalyticVI(),10,optimizer=false)
    train!(m)
    ELBO(m)
end

train_and_ELBO(0.5)

using ForwardDiff
W = rand(100,3)
reshape(ForwardDiff.jacobian(x->kernelmatrix(SqExponentialKernel(x),W,obsdim=1),[0.5,0.2,0.1]),100,100)
using Zygote
Zygote.gradient(train_and_ELBO,0.5)
Zygote.gradient(k->logdet(kernelmatrix(k,reshape(X,:,1),obsdim=1)),SqExponentialKernel(0.1))[1][:transform][:s][][:x]
Zygote.gradient(k->logdet(kernelmatrix(k,reshape(X,:,1),obsdim=1)),SqExponentialKernel([0.1]))[1][:transform][:s]
