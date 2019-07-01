using Pkg
pkg"add AugmentedGaussianProcesses"
using AugmentedGaussianProcesses
const AGP = AugmentedGaussianProcesses
using Distributions
using StatsBase, Distances
using Random: seed!
using BenchmarkTools
using ProfileView, Profile, Traceur
using MLDataUtils
seed!(42)

N_data = 100
N_dim=2
N_class = 10
N_test = 50
minx=-5.0
maxx=5.0
noise = 1.0

for c in 1:N_class
    global centers = rand(Uniform(minx,maxx),N_class,N_dim)
    global variance = 1/N_class*ones(N_class)#rand(Gamma(1.0,0.5),150)
end

X = zeros(N_data,N_dim)
y = sample(1:N_class,N_data)
for i in 1:N_data
    X[i,:] = rand(MvNormal(centers[y[i],:],variance[y[i]]))
end
(X,y),(X_test,y_test) = splitobs((X,y),at=0.66,obsdim=1)

function initial_lengthscale(X)
    D = pairwise(SqEuclidean(),X,dims=1)
    return median([D[i,j] for i in 2:size(D,1) for j in 1:(i-1)])
end
l = sqrt(initial_lengthscale(X))


kernel = AugmentedGaussianProcesses.RBFKernel([l],dim=N_dim)

model = SVGP(X,y,kernel,LogisticSoftMaxLikelihood(),AnalyticSVI(10),10)

train!(model,iterations=1)

##Precompile functions
AGP.computeMatrices!(model)
AGP.update_hyperparameters!(model)
AGP.update_parameters!(model)
AGP.computeMatrices!(model)

##Benchmark time estimation
@btime AGP.update_parameters!($model);
@btime AGP.update_hyperparameters!($model);
@btime AGP.natural_gradient!($model);
# @btime AGP.natural_gradient_old!($model);
AGP.computeMatrices!(model)

##Profiling
@code_warntype map!(AGP.∇η₂,model.inference.∇η₂,model.likelihood.θ,fill(model.inference.ρ,model.nLatent),model.κ,model.invKmm,model.η₂)

@profiler AGP.update_parameters!(model);
@profiler repeat(AGP.natural_gradient!(model),1000);
@profiler repeat(AGP.∇η₂.(model.likelihood.θ,fill(model.inference.ρ,model.nLatent),model.κ,model.invKmm,model.η₂),1000);
@profiler AGP.update_hyperparameters!(model)
