using AugmentedGaussianProcesses
using Distributions,LinearAlgebra
using Random: seed!; seed!(42)
using Plots; pyplot()


N = 200;#Number of training points
N_test = 1000 # Number of points for predictions
X = sort(rand(N)) # Training points
shift = 0.1*(maximum(X)-minimum(X))
X_test = collect(range(minimum(X)-shift,length=N_test,stop=maximum(X)+shift))
## Noise parameters
σ = 5e-1 #Gaussian noise
ν = 4.0; #Degrees of freedom for Student T
β = 5e-1; # Shape for Laplace
mu0 = 0.3; # Constant for heteroscedastic gp
logit(x) =  1.0./(1.0.+exp.(-x))
## Kernel
k = RBFKernel(0.1) # Squared exponential kernel
K = Symmetric(kernelmatrix(reshape(X,:,1),k)+1e-9*Diagonal(I,N)) #Kernel matrix
f = rand(MvNormal(zeros(N),K)) #Random GP sampled from the prior
g = rand(MvNormal(mu0*ones(N),K)) #Random GP sampled from the prior for Heteroscedasticity
y_gaussian = f .+ rand(Normal(0,σ),N)
y_studentt = f .+ rand(TDist(ν),N)
y_laplace = f .+ rand(Laplace(0,β),N)
y_hetero = f .+ randn(N).*sqrt.(inv.(logit.(g)))

gpmodel= GP(X,y_gaussian,k,noise=σ^2);
train!(gpmodel,iterations=50)#,callback=intplot)
gppred,gppred_cov = proba_y(gpmodel,X_test)

stumodel = VGP(X,y_studentt,k,StudentTLikelihood(ν),AnalyticVI());
train!(stumodel,iterations=50)#,callback=intplot)
stupred,stupred_cov = proba_y(stumodel,X_test)

lapmodel = VGP(X,y_laplace,k,LaplaceLikelihood(β),AnalyticVI());
train!(lapmodel,iterations=50)#,callback=intplot)
lappred,lappred_cov = proba_y(lapmodel,X_test)

hetmodel = VGP(X,y_hetero,k,HeteroscedasticLikelihood(k,convert(PriorMean,mu0)),AnalyticVI())
train!(hetmodel,iterations=50)
hetpred,hetpred_cov = proba_y(hetmodel,X_test)

## Plotting
lw = 3.0 #Width of the lines
nsig = 2 #Number of sigmas for the uncertainty
falpha = 0.3

pstudent = plot(X,y_studentt,t=:scatter,lab="")
plot!(X_test,stupred,title="StudentT Regression",lw=lw,color=1,lab="")
plot!(X,f,color=:black,lw=1.0,lab="")
plot!(X,f.+nsig*sqrt(ν/(ν-2)),lab="",color=:black,lw=1.0,linestyle=:dash)
plot!(X,f.-nsig*sqrt(ν/(ν-2)),lab="",color=:black,lw=1.0,linestyle=:dash)
maxlims = ylims(pstudent)
plot!(X_test,stupred.+ nsig  .* sqrt.(stupred_cov),linewidth=0.0,
    fillrange=stupred .- nsig .* sqrt.(stupred_cov),
    fillalpha=falpha,
    fillcolor=1,
    label="")
ylims!(maxlims)

pgauss = plot(X,y_gaussian,t=:scatter,lab="")
plot!(X_test,gppred,title="Gaussian Regression",lw=lw,color=1,lab="")
plot!(X,f,color=:black,lw=1.0,lab="")
plot!(X,f.+nsig*σ,lab="",color=:black,lw=1.0,linestyle=:dash)
plot!(X,f.-nsig*σ,lab="",color=:black,lw=1.0,linestyle=:dash)
plot!(X_test,gppred .+ nsig .* sqrt.(gppred_cov),linewidth=0.0,
    fillrange=gppred .- nsig  .* sqrt.(gppred_cov),
    fillalpha=falpha,
    fillcolor=1,
    label="")
ylims!(maxlims)

plaplace = plot(X,y_laplace,t=:scatter,lab="")
plot!(X_test,lappred,title="Laplace Regression",lw=lw,color=1,lab="")
plot!(X,f,color=:black,lw=1.0,lab="")
plot!(X,f.+nsig*β*sqrt(2),lab="",color=:black,lw=1.0,linestyle=:dash)
plot!(X,f.-nsig*β*sqrt(2),lab="",color=:black,lw=1.0,linestyle=:dash)
plot!(X_test,lappred.+ nsig  .* sqrt.(lappred_cov),linewidth=0.0,
    fillrange=lappred .- nsig .* sqrt.(lappred_cov),
    fillalpha=falpha,
    fillcolor=1,
    label="")
ylims!(maxlims)

phetero = plot(X,y_hetero,t=:scatter,lab="")
plot!(X_test,lappred,title="Heteroscedastic Regression",lw=lw,color=1,lab="")
plot!(X,f,color=:black,lw=1.0,lab="")
plot!(X,f.+nsig*sqrt.(inv.(logit.(g))),lab="",color=:black,lw=1.0,linestyle=:dash)
plot!(X,f.-nsig*sqrt.(inv.(logit.(g))),lab="",color=:black,lw=1.0,linestyle=:dash)
plot!(X_test,hetpred.+ nsig  .* sqrt.(hetpred_cov),linewidth=0.0,
    fillrange=hetpred .- nsig .* sqrt.(hetpred_cov),
    fillalpha=falpha,
    fillcolor=1,
    label="")
ylims!(maxlims)

default(legendfontsize=10.0,xtickfontsize=10.0,ytickfontsize=10.0)
p=plot(pgauss,pstudent,plaplace,phetero)
display(p)
savefig(p,joinpath(@__DIR__,"Regression.png"))
