using AugmentedGaussianProcesses
using Distributions, LinearAlgebra
using Random: seed!;
seed!(42);
using Plots;
pyplot();

N = 200;#Number of training points
N_test = 1000 # Number of points for predictions
X = sort(rand(N)) # Training points
shift = 0.1 * (maximum(X) - minimum(X))
X_test = collect(range(minimum(X) - shift; length=N_test, stop=maximum(X) + shift))
## Noise parameters
σ = 5e-1 #Gaussian noise
ν = 4.0; #Degrees of freedom for Student T
β = 5e-1; # Shape for Laplace
mu0 = 0.3; # Constant for heteroscedastic gp
## Kernel
k = SqExponentialKernel() ∘ ScaleTransform(10.0) # Squared exponential kernel
K = kernelmatrix(k, X) + 1e-9 * I #Kernel matrix
f = rand(MvNormal(K)) #Random GP sampled from the prior
g = rand(MvNormal(mu0 * ones(N), K)) #Random GP sampled from the prior for Heteroscedasticity
y_gaussian = f .+ rand(Normal(0, σ), N)
y_studentt = f .+ rand(TDist(ν), N)
y_laplace = f .+ rand(Laplace(0, β), N)
y_hetero = f .+ randn(N) .* sqrt.(inv.(AGP.logistic.(g)))

gpmodel = GP(X, y_gaussian, k; noise=σ^2);
train!(gpmodel, 50)
gppred, gppred_cov = proba_y(gpmodel, X_test)

stumodel = VGP(X, y_studentt, k, StudentTLikelihood(ν), AnalyticVI());
train!(stumodel, 50)#,callback=intplot)
stupred, stupred_cov = proba_y(stumodel, X_test)

lapmodel = VGP(X, y_laplace, k, LaplaceLikelihood(β), AnalyticVI());
train!(lapmodel, 50)#,callback=intplot)
lappred, lappred_cov = proba_y(lapmodel, X_test)

hetmodel = VGP(X, y_hetero, k, HeteroscedasticLikelihood(), AnalyticVI())
train!(hetmodel, 100)
hetpred, hetpred_cov = proba_y(hetmodel, X_test)

## Plotting
lw = 4.0 #Width of the lines
lwt = 2.0
nsig = 2 #Number of sigmas for the uncertainty
falpha = 0.3
pstudent = scatter(X, y_studentt; t=:scatter, lab="")
plot!(
    X_test,
    stupred;
    ribbon=nsig .* sqrt.(stupred_cov),
    title="StudentT Regression",
    lw=lw,
    color=1,
    lab="",
    fillalpha=falpha,
    linecolor=2,
)
plot!(X, f; color=:black, lw=lwt, lab="")
plot!(X, f .+ nsig * sqrt(ν / (ν - 2)); lab="", color=:black, lw=lwt, linestyle=:dash)
plot!(X, f .- nsig * sqrt(ν / (ν - 2)); lab="", color=:black, lw=lwt, linestyle=:dash)
# ylims!(maxlims)

pgauss = scatter(X, y_gaussian; lab="")
plot!(
    X_test,
    gppred;
    ribbon=nsig * gppred_cov,
    title="Gaussian Regression",
    lw=lw,
    color=1,
    lab="",
    fillalpha=falpha,
    linecolor=2,
)
plot!(X, f; color=:black, lw=lwt, lab="")
plot!(X, f .+ nsig * σ; lab="", color=:black, lw=lwt, linestyle=:dash)
plot!(X, f .- nsig * σ; lab="", color=:black, lw=lwt, linestyle=:dash)

plaplace = scatter(X, y_laplace; lab="")
plot!(
    X_test,
    lappred;
    ribbon=nsig * sqrt.(lappred_cov),
    title="Laplace Regression",
    lw=lw,
    color=1,
    lab="",
    fillalpha=falpha,
    linecolor=2,
)
plot!(X, f; color=:black, lw=lwt, lab="")
plot!(X, f .+ nsig * β * sqrt(2); lab="", color=:black, lw=lwt, linestyle=:dash)
plot!(X, f .- nsig * β * sqrt(2); lab="", color=:black, lw=lwt, linestyle=:dash)

phetero = plot(X, y_hetero; t=:scatter, lab="")
plot!(
    X_test,
    lappred;
    ribbon=nsig .* sqrt.(hetpred_cov),
    title="Heteroscedastic Regression",
    lw=lw,
    color=1,
    lab="",
    fillalpha=falpha,
    linecolor=2,
)
plot!(X, f; color=:black, lw=lwt, lab="")
plot!(
    X,
    f .+ nsig * sqrt.(inv.(AGP.logistic.(g)));
    lab="",
    color=:black,
    lw=lwt,
    linestyle=:dash,
)
plot!(
    X,
    f .- nsig * sqrt.(inv.(AGP.logistic.(g)));
    lab="",
    color=:black,
    lw=lwt,
    linestyle=:dash,
)

# default(legendfontsize=10.0,xtickfontsize=10.0,ytickfontsize=10.0)
p = plot(pgauss, pstudent, plaplace, phetero; ylims=(-5.0, 5.0))
display(p)
savefig(p, joinpath(@__DIR__, "Regression.png"))
