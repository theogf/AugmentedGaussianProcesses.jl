using ForwardDiff
# using Zygote
using AugmentedGaussianProcesses
using AugmentedGaussianProcesses.KernelModule
using MLKernels
using LinearAlgebra
using Distributions
using BenchmarkTools
using Expectations

N_data = 1000;N_test = 20
N_indpoints = 80; N_dim = 2
noise = 0.1
minx=-5.0; maxx=5.0
function latent(x)
    return sqrt.(x[:,1].^2+x[:,2].^2)
end
X = rand(N_data,N_dim)*(maxx-minx).+minx
x_test = range(minx,stop=maxx,length=N_test)
X_test = hcat([j for i in x_test, j in x_test][:],[i for i in x_test, j in x_test][:])
y = floor.(Int64,latent(X+rand(Normal(0,noise),N_data,N_dim)))
y_test = floor.(Int64,latent(X_test))
A = rand(N_data,N_data)
K = length(unique(y))

function tracekernel(sigma)
     k= RBFKernel(sigma[1],dim=N_dim)
     K = kernelmatrix(X,k)
     tr(A*K)
end

# tracekernel(3.0)
# tracekernel'(3.0)
# ForwardDiff.gradient(tracekernel,[3.0])
m = SparseMultiClass(X,y,Autotuning=true,m=100,kernel=RBFKernel(1.0))
train!(m,iterations=10)
### TEST 1 with matrix precomputation

function compute_grad(model)
    f_l,f_v = AugmentedGaussianProcesses.hyperparameter_gradient_function(model)
    matrix_derivatives =[[KernelModule.kernelderivativematrix(model.Z[kiter],model.kernel[kiter]),
    KernelModule.kernelderivativematrix(model.X[model.inference.MBIndices,:],model.Z[kiter],model.kernel[kiter]),
    KernelModule.kernelderivativediagmatrix(model.X[model.inference.MBIndices,:],model.kernel[kiter])] for kiter in 1:model.nLatent]
    grads_l = map(AugmentedGaussianProcesses.compute_hyperparameter_gradient,model.kernel,[f_l for _ in 1:model.nLatent],matrix_derivatives,1:model.nLatent)
    grads_v = map(f_v,model.kernel,1:model.nLatent)
    return grads_l,grads_v
end

vars = ones(m.nLatent*2)
@btime m.Knm .= broadcast((Z,kernel)->KernelModule.kernelmatrix(m.X[m.inference.MBIndices,:],Z,kernel),m.Z,m.kernel);

@profiler broadcast((Z::AbstractMatrix{<:Real},kernel::KernelModule.Kernel)->KernelModule.kernelmatrix(m.X[m.inference.MBIndices,:],Z,kernel),m.Z,m.kernel)
KernelModule.kernelmatrix(m.X[m.inference.MBIndices,:],m.Z[1],m.kernel[1])
function compute_grad_AD(k_var)
    kernel = [KernelModule.RBFKernel(k_var[i],variance=k_var[i+K]) for i in 1:K]
    m = SparseMultiClass(X,y,Autotuning=true,m=100,kernel=kernel)
    m.inference.HyperParametersUpdated = true
    AugmentedGaussianProcesses.computeMatrices!(m)
    ELBO(m)
end

compute_grad_AD(vars)
compute_grad(m)

@btime compute_grad($m)
conf = ForwardDiff.GradientConfig(compute_grad_AD,vars)
ForwardDiff.gradient(compute_grad_AD,vars)
@btime ForwardDiff.gradient($compute_grad_AD,$vars,$conf)

# function ELBO(model::SparseXGPC,blah::Bool)
# function ADELBO(kernelparams_and_indpoints::AbstractArray)
#     kernelparams=kernelparams_and_indpoints[1:model.nDim+1]
#     model.inducingPoints=reshape(kernelparams_and_indpoints[model.nDim+2:end],(model.m,model.nDim))
#     # println(kernelparams)
#     model.kernel = RBFKernel(kernelparams[1:model.nDim],dim=model.nDim,variance=kernelparams[end])
#     model.HyperParametersUpdated =true
#     AugmentedGaussianProcesses.computeMatrices!(model)
#     AugmentedGaussianProcesses.ELBO(model)
# end

# # tracekernel'(3.0)
# grads = ForwardDiff.gradient(ADELBO,vcat(getlengthscales(model.kernel),getvariance(model.kernel),model.inducingPoints[:]))
# vcat(getlengthscales(model.kernel),getvariance(model.kernel),model.inducingPoints[:])
# # end
# ADELBO([[getlengthscales(model.kernel),getvariance(model.kernel)],model.inducingPoints])
# AugmentedGaussianProcesses.computeMatrices!(model)
# ELBO(model)
# typeof(model.kernel)
# model.Kmm = Symmetric(kernelmatrix(X,model.kernel)+1.0e-7*I)
#
logit(x) = 1.0./(1.0.+exp.(-x))
sigsoftmax(i,f) = logit(f[i])/sum(logit.(f))
softmax(i,f) = exp(f[i])/sum(exp.(f))
invKmm = copy(m.invKmm)
K = length(invKmm)
Nm = length(m.μ[1])
ym = sample(1:K,Nm,replace=true)
function sig_elbo(args)
    tot = 0
    μ = reshape(args[1:Nm*K],K,Nm)
    Σ = reshape(args[Nm*K+1:end],K,Nm)
    p = MvNormal(μ[:,1],sqrt.(Σ[:,1]))
    x = zero(μ[:,1])
    for i in 1:Nm
        # tot += sum(sigsoftmax(ym[i],rand(K)) for _ in 1:200)
        p = MvNormal(μ[:,i],sqrt.(Σ[:,i]))
        tot += sum(sigsoftmax(ym[i],Distributions._rand!(p,x)) for _ in 1:200)
    end
    return tot +
    sum(0.5*(sum(invKmm[i].*(Σ[i]+μ[i]*transpose(μ[i])))-logdet(Σ[i])-logdet(invKmm[i])) for i in 1:K)
end

function logit(x)
    return 1.0./(1.0.+exp.(-x))
end

function mod_soft_max(x,i::Integer)
    return logit(x[i])/sum(logit(x))
end
function mod_soft_max(x)
    return logit(x)./sum(logit(x))
end
function grad_mod(s,σ,i)
    return [(δ(i,j)-s[i])*s[j]*(1-σ[j]) for j in 1:length(s)]
end
function hessian_mod(s,σ,i)
end
function δ(i::Integer,j::Integer)
    i == j ? 1.0 : 0.0
end
a=rand(10)
logit(a[1])
mod_soft_max(a,1)
ForwardDiff.gradient(x->mod_soft_max(x,1),a)
grad_mod(mod_soft_max(a),logit(a),1)
