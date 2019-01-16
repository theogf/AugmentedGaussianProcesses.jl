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
noise = 2.0
minx=-5.0; maxx=5.0
function latent(x)
    return 0.5*x[:,1].*sin.(1.0*x[:,2])
end
X = rand(N_data,N_dim)*(maxx-minx).+minx
x_test = range(minx,stop=maxx,length=N_test)
X_test = hcat([j for i in x_test, j in x_test][:],[i for i in x_test, j in x_test][:])
y = floor.(Int64,latent(X)+rand(Normal(0,noise),size(X,1)))
y_test = floor.(Int64,latent(X_test))
A = rand(N_data,N_data)


function tracekernel(sigma)
     k= RBFKernel(sigma[1],dim=N_dim)
     K = kernelmatrix(X,k)
     tr(A*K)
end

# tracekernel(3.0)
# tracekernel'(3.0)
# ForwardDiff.gradient(tracekernel,[3.0])
m = SparseMultiClass(X,y,Autotuning=true,m=100,kernel=RBFKernel(ones(N_dim)))
m.train(iterations=10)
### TEST 1 with matrix precomputation

function compute_grad(model)
    f_l,f_v = AugmentedGaussianProcesses.hyperparameter_gradient_function(model)
    matrix_derivatives =[[KernelModule.kernelderivativematrix(model.inducingPoints[kiter],model.kernel[kiter]),
    KernelModule.kernelderivativematrix(model.X[model.MBIndices,:],model.inducingPoints[kiter],model.kernel[kiter]),
    KernelModule.kernelderivativediagmatrix(model.X[model.MBIndices,:],model.kernel[kiter])] for kiter in model.KIndices]
    grads_l = map(AugmentedGaussianProcesses.compute_hyperparameter_gradient,model.kernel[model.KIndices],[f_l for _ in 1:model.nClassesUsed],matrix_derivatives,model.KIndices,1:model.nClassesUsed)
    grads_v = map(f_v,model.kernel[model.KIndices],model.KIndices,1:model.nClassesUsed)
    return grads_l,grads_v
end

vars = ones(length(m.altkernel)*2)

function compute_grad_AD(k_var)
    m.altkernel = [MLKernels.SquaredExponentialKernel(1.0./k_var[1:m.K].^2) for _ in 1:m.K]
    m.altvar = k_var[m.K+1:end]
    m.HyperParametersUpdated = true
    AugmentedGaussianProcesses.computeMatrices!(m)
    ELBO(m)
end

compute_grad_AD(vars)

@btime compute_grad($m)
conf = ForwardDiff.GradientConfig(compute_grad_AD,vars)
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


sig_elbo(vcat(vcat(m.μ...),vcat(diag.(m.Σ)...)))
ForwardDiff.gradient(sig_elbo,vcat(vcat(m.μ...),vcat(diag.(m.Σ)...)))
m.μ


function hessian_softmax(s::AbstractVector{<:Real},grad::AbstractVector{<:Real},i::Integer,stot::Real=0.0)
    hessian = zeros(m,m)
    for j in 1:m
        for k in 1:m
            hessian[j,k] = s[i]*((δ(i,k)-s[k])*(δ(i,j)-s[j])-s[j]*(δ(j,k)-s[k]))
        end
    end
    return hessian
end
