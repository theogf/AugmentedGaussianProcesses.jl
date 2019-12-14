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
N_dim=10
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
y = rand(1:N_class,N_data)
for i in 1:N_data
    X[i,:] = rand(MvNormal(centers[y[i],:],variance[y[i]]))
end
(X,y),(X_test,y_test) = splitobs((X,y),at=0.66,obsdim=1)

function initial_lengthscale(X)
    D = pairwise(SqEuclidean(),X,dims=1)
    return median([D[i,j] for i in 2:size(D,1) for j in 1:(i-1)])
end
l = sqrt(initial_lengthscale(X))
ll = l*ones(N_dim)

kernel = SqExponentialKernel(ll)

model = VGP(X,y,kernel,LogisticSoftMaxLikelihood(),AnalyticVI())

train!(model,1)

##Precompile functions
AGP.computeMatrices!(model)
AGP.update_hyperparameters!(model)
AGP.update_parameters!(model)
AGP.computeMatrices!(model)

##Benchmark time estimation
# @btime AGP.update_parameters!($model);
# @btime AGP.update_hyperparameters!($model);
# @btime AGP.natural_gradient_old!($model);
AGP.computeMatrices!(model)

##Profiling

@profiler AGP.update_parameters!(model);
@profiler repeat(AGP.∇η₂.(model.likelihood.θ,fill(model.inference.ρ,model.nLatent),model.κ,model.invKmm,model.η₂),1000);
Profile.clear()
@profiler AGP.update_hyperparameters!(model)
@profview AGP.update_hyperparameters!(model)
f_l,f_v,f_μ₀ = AGP.hyperparameter_gradient_function(model.f[1],X)
@profiler AGP.kernelderivative(model.f[1].kernel,X)
@btime AGP.kernelderivative($(model.f[1].kernel),$X);
Jz = first(AGP.kernelderivative((model.f[1].kernel),X))[2]
Jf = reshape(ForwardDiff.jacobian(x->kernelmatrix(SqExponentialKernel(x),X,obsdim=1),ll),size(X,1),size(X,1),2)
@btime AGP.kernelderivative($(model.f[1].kernel),$X);
@btime reshape(ForwardDiff.jacobian(x->kernelmatrix(SqExponentialKernel(x),X,obsdim=1),ll),size(X,1),size(X,1),2);
using ForwardDiff

f_l,f_v,f_μ₀ = AGP.hyperparameter_gradient_function(model.f[1],X)
gp = model.f[1]
ps = Flux.params(gp.kernel)
function old_method(gp,X,f_l)
    Jnn = reshape(ForwardDiff.jacobian(x->kernelmatrix(SqExponentialKernel(x),X,obsdim=1),ll),size(X,1),size(X,1),N_dim);
    mapslices(f_l,Jnn,dims=[1,2])
end
vec(old_method(gp,X,f_l))
AGP.∇L_ρ(gp,X,f_l)[first(ps)][:]
@btime AGP.∇L_ρ(gp,X,f_l)
@btime old_method(gp,X,f_l)
