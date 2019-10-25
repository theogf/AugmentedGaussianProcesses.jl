using AugmentedGaussianProcesses; const AGP= AugmentedGaussianProcesses
using MLDataUtils
using KernelFunctions

x,f = noisy_function(sinc,range(-3,3,length=100))
y = sign.(f)

M = VGP(x,y,SqExponentialKernel(),LogisticLikelihood(),AnalyticVI(),optimizer=false)
train!(M)

m = SVGP(x,y,SqExponentialKernel(),LogisticLikelihood(),AnalyticVI(),10,optimizer=false,verbose=3)
train!(m,100)
ELBO(m)


pred_F = predict_f(M,x,covf=false)
pred_f = predict_f(m,reshape(x,:,1),covf=false)
pred_X = predict_y(M,x)
pred_x = predict_y(m,x)
using Plots
scatter(x,y)
scatter!(x,pred_x)
scatter!(x,pred_X)
scatter!(AGP.get_X(m)[:],zeros(length(AGP.get_X(m))))
plot!(x,pred_f)
plot!(x,pred_F)
