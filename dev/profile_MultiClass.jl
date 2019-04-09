# using Pkg
# pkg"add AugmentedGaussianProcesses"
using AugmentedGaussianProcesses
using Distributions
using StatsBase, Distances
using Random: seed!
using BenchmarkTools
using PyCall
using ProfileView, Profile, Traceur
const AGP = AugmentedGaussianProcesses
seed!(42)

@pyimport sklearn.model_selection as sp
N_data = 2000
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
X,X_test,y,y_test = sp.train_test_split(X,y,test_size=0.33)

function initial_lengthscale(X)
    D = pairwise(SqEuclidean(),X')
    return median([D[i,j] for i in 2:size(D,1) for j in 1:(i-1)])
end
l = sqrt(initial_lengthscale(X))


kernel = AugmentedGaussianProcesses.RBFKernel([l],dim=N_dim)

model = SVGP(X,y,kernel,LogisticSoftMaxLikelihood(),AnalyticVI(),500,Autotuning=true,atfrequency=1)

train!(model,iterations=1)

##Precompile functions
AGP.computeMatrices!(model)
AGP.update_hyperparameters!(model)
AGP.computeMatrices!(model)

##Benchmark time estimation
@btime AGP.update_hyperparameters!(model)
AGP.computeMatrices!(model)
Profile.clear()

##Profiling

@profiler AGP.update_hyperparameters!(model)
