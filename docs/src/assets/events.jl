using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
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
y = rand.(Poisson.(10*AGP.logistic.(f)))
function intplot(model,iter)
    p=plot()
    plot!(X,y,t=:scatter);
    pred=(proba_y(model,X_test).-0.5).*2; plot!(X_test,pred);
    # if minimum(pred[2500])
    title!("Iteration : $iter, var : $(AugmentedGaussianProcesses.getvariance(model.kernel[1]))")
    # fstar = model.fstar(X_test,covf=false); plot!(X_test,logit(fstar));
    display(p);
end

plot(X,y,t=:scatter)
poissonmodel= VGP(X,y,k,PoissonLikelihood(),AnalyticVI());
train!(logitmodel,iterations=300)#,callback=intplot)
poissonpred = predict_y(logitmodel,X_test)
plot!(X_test,poissonpred)
negbinmodel = VGP(X,y,k,NegBinomialLikelihood(10),AnalyticVI());
train!(negbinmodel,iterations=50)#,callback=intplot)
negbinpred = predict_y(negbinmodel,X_test)
# svmpred2 = (AugmentedGaussianProcesses.svmpredictproba(svmmodel,X_test).-0.5)*2
# svmfstar,covfstar = svmmodel.fstar(X_test,covf=true)
# y1 = AugmentedGaussianProcesses.svmlikelihood(svmfstar)
# y2 = AugmentedGaussianProcesses.svmlikelihood(-svmfstar)
# plot(X_test,y1+y2)
##
default(legendfontsize=14.0,xtickfontsize=10.0,ytickfontsize=10.0,ylims = (-0.1,20))
p = plot(X,y,t=:scatter,lab="Training Points")
plot!(X_test,poissonpred,lab="Poisson Expectation",lw=7.0)
plot!(X_test,negbinpred,lab="Negative Binomial Expectation",lw=7.0,legend=:topright)

display(p)
savefig(p,String(@__DIR__)*"/Events.png")
findmin(logitpred)
# a,b = logitmodel.fstar(X_test)
# using Expectations, QuadGK, BenchmarkTools

# @btime begin; for i in 1:N_test; d = Normal(a[i],sqrt(max(b[i],1e-8))); expectation(logit,d); end; end;
# @btime begin; for i in 1:N_test; d = Normal(a[i],sqrt(max(b[i],1e-8))); quadgk(x->pdf(d,x)*logit(x),-Inf,Inf); end; end;
