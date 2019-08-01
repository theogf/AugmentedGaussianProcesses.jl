using AugmentedGaussianProcesses
using Distributions,LinearAlgebra
using Random: seed!; seed!(123)
using Plots; gr();

N = 100; noise = 1e-1
N_test = 10000
X = sort(rand(N))
X_test = collect(range(0,length=N_test,stop=1))
k = RBFKernel(0.1)
K = Symmetric(kernelmatrix(reshape(X,:,1),k)+noise*Diagonal(I,N))
f = rand(MvNormal(zeros(N),K))
y = sign.(f)
function intplot(model,iter)
    p=plot()
    plot!(X,y,t=:scatter);
    pred=(proba_y(model,X_test).-0.5).*2; plot!(X_test,pred);
    # if minimum(pred[2500])
    title!("Iteration : $iter, var : $(AugmentedGaussianProcesses.getvariance(model.kernel[1]))")
    # fstar = model.fstar(X_test,covf=false); plot!(X_test,logit(fstar));
    display(p);
end
logit(x) =  1.0./(1.0.+exp.(-x))

plot(X,f,t=:scatter)
plot!(X,y,t=:scatter)
logitmodel= VGP(X,y,k,LogisticLikelihood(),AnalyticVI());
train!(logitmodel,iterations=300)#,callback=intplot)
logitpred = (proba_y(logitmodel,X_test).-0.5).*2
plot!(X_test,logitpred)
svmmodel = VGP(X,y,k,BayesianSVM(),AnalyticVI());
train!(svmmodel,iterations=50)#,callback=intplot)
svmpred = (proba_y(svmmodel,X_test).-0.5).*2
# svmpred2 = (AugmentedGaussianProcesses.svmpredictproba(svmmodel,X_test).-0.5)*2
# svmfstar,covfstar = svmmodel.fstar(X_test,covf=true)
# y1 = AugmentedGaussianProcesses.svmlikelihood(svmfstar)
# y2 = AugmentedGaussianProcesses.svmlikelihood(-svmfstar)
# plot(X_test,y1+y2)
##
default(legendfontsize=14.0,xtickfontsize=10.0,ytickfontsize=10.0)
p = plot(X,y,t=:scatter,lab="Training Points")
plot!(X_test,logitpred,lab="Logistic Prediction",lw=7.0)
plot!(X_test,svmpred,lab="BayesianSVM Prediction",lw=7.0,legend=:right)

display(p)
savefig(p,String(@__DIR__)*"/Classification.png")
findmin(logitpred)
# a,b = logitmodel.fstar(X_test)
# using Expectations, QuadGK, BenchmarkTools

# @btime begin; for i in 1:N_test; d = Normal(a[i],sqrt(max(b[i],1e-8))); expectation(logit,d); end; end;
# @btime begin; for i in 1:N_test; d = Normal(a[i],sqrt(max(b[i],1e-8))); quadgk(x->pdf(d,x)*logit(x),-Inf,Inf); end; end;
