using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using LinearAlgebra, Distributions, Plots
using BenchmarkTools
b = 2.0
C()=1/(2b)
g(y) = 0.0
α(y) = y^2
β(y) = 2*y
γ(y) = 1.0
φ(r) = exp(-sqrt(r)/b)
∇φ(r) = -exp(-sqrt(r)/b)/(2*b*sqrt(r))
txt = AGP.@augmodel("NewLaplace","Regression",C,g,α,β,γ,φ,∇φ)
# NewLaplaceLikelihood() |> display
N = 500
σ = 1.0
X = sort(rand(N,1),dims=1)
K = kernelmatrix(X,RBFKernel(0.1))+1e-4*I
L = Matrix(cholesky(K).L)
y_true = rand(MvNormal(K))
y = y_true+randn(length(y_true))*2
p = scatter(X[:],y,lab="data")
m = VGP(X,y,RBFKernel(0.5),NewLaplaceLikelihood(),AnalyticVI(),optimizer=false)
train!(m,iterations=100)
y_p, sig_p = proba_y(m,collect(0:0.01:1))

m2 = VGP(X,y,RBFKernel(0.5),LaplaceLikelihood(b),AnalyticVI(),optimizer=false)
train!(m2,iterations=100)
y_p2, sig_p2 = proba_y(m2,collect(0:0.01:1))

plot!(X,y_true,lab="truth")

plot!(collect(0:0.01:1),y_p,lab="Auto Laplace")
plot!(collect(0:0.01:1),y_p2,lab="Classic Laplace")


@btime train!($m,iterations=1)
@btime train!($m2,iterations=1)
