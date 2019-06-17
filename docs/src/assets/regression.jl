using AugmentedGaussianProcesses
using Distributions,LinearAlgebra
using Random: seed!; seed!(123)
using Plots; pyplot()

N = 200; noise = 5e-1
N_test = 10000
X = sort(randn(N))
expand = 1.1
X_test = collect(range(expand*minimum(X),length=N_test,stop=expand*maximum(X)))
k = RBFKernel(0.1)
K = Symmetric(kernelmatrix(reshape(X,:,1),k)+1e-9*Diagonal(I,N))
K_noise = Symmetric(kernelmatrix(reshape(X,:,1),k)+1e-9*Diagonal(I,N))
f = rand(MvNormal(zeros(N),K))
# g = rand(MvNormal(zeros))
y = f .+ rand(Normal(0,noise),N)
function intplot(model,iter)
    p=plot()
    plot!(X,y,t=:scatter);
    pred=model.predict(X_test); plot!(X_test,pred);
    title!("var : $(AugmentedGaussianProcesses.getvariance(model.kernel))")
    # fstar = model.fstar(X_test,covf=false); plot!(X_test,logit(fstar));
    display(p);
end
logit(x) =  1.0./(1.0.+exp.(-x))

plot(X,f,t=:scatter)
gpmodel= GP(X,y,k,Autotuning=true,noise=noise);
train!(gpmodel,iterations=50)#,callback=intplot)
gppred,gppred_cov = proba_y(gpmodel,X_test)
plot!(X_test,gppred)
stumodel = VGP(X,y,k,StudentTLikelihood(5.0),AnalyticVI(),Autotuning=true);
train!(stumodel,iterations=50)#,callback=intplot)
stupred,stupred_cov = proba_y(stumodel,X_test)
lapmodel = VGP(X,y,k,LaplaceLikelihood(),AnalyticVI(),Autotuning=true);
train!(lapmodel,iterations=50)#,callback=intplot)
lappred,lappred_cov = proba_y(lapmodel,X_test)
hetmodel = VGP(X,y,k,HeteroscedasticLikelihood(RBFKernel(0.1)),AnalyticVI(),Autotuning=true)
train!(hetmodel,iterations=50)
hetpred,hetpred_cov = proba_y(hetmodel,X_test)

##
lw = 3.0
nsig = 2
falpha = 0.3
p2 = plot(X,y,t=:scatter,lab="")
plot!(X_test,stupred,title="StudentT Regression",lw=lw,color=1,lab="")
plot!(X_test,stupred.+ nsig  .* sqrt.(stupred_cov),linewidth=0.0,
    fillrange=stupred .- nsig .* sqrt.(stupred_cov),
    fillalpha=falpha,
    fillcolor=1,
    label="")
p1 = plot(X,y,t=:scatter,lab="")
plot!(X_test,gppred,title="Gaussian Regression",lw=lw,color=1,lab="")
plot!(X_test,gppred .+ nsig .* sqrt.(gppred_cov),linewidth=0.0,
    fillrange=gppred .- nsig  .* sqrt.(gppred_cov),
    fillalpha=falpha,
    fillcolor=1,
    label="")
ylims!(ylims(p2))
p3 = plot(X,y,t=:scatter,lab="")
plot!(X_test,lappred,title="Laplace Regression",lw=lw,color=1,lab="")
plot!(X_test,lappred.+ nsig  .* sqrt.(lappred_cov),linewidth=0.0,
    fillrange=lappred .- nsig .* sqrt.(lappred_cov),
    fillalpha=falpha,
    fillcolor=1,
    label="")
ylims!(ylims(p2))
p4 = plot(X,y,t=:scatter,lab="")
plot!(X_test,lappred,title="Heteroscedastic Regression",lw=lw,color=1,lab="")
plot!(X_test,hetpred.+ nsig  .* sqrt.(hetpred_cov),linewidth=0.0,
    fillrange=hetpred .- nsig .* sqrt.(hetpred_cov),
    fillalpha=falpha,
    fillcolor=1,
    label="")
ylims!(ylims(p2))
default(legendfontsize=10.0,xtickfontsize=10.0,ytickfontsize=10.0)
p=plot(p1,p2,p3,p4)
display(p)
savefig(p,joinpath(@__DIR__,"Regression.png"))
