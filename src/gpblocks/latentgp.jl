### Latent models ####
include("prior.jl")
include("posterior.jl")

## Exact Gaussian Process
struct LatentGP{T,Tpr<:GPPrior,Tpo<:Posterior{T},O} <: AbstractLatent{T,Tpr,Tpo}
    prior::Tpr
    post::Tpo
    opt::O
end

function LatentGP(T::DataType, dim::Int, kernel::Kernel, mean::PriorMean, opt)
    return LatentGP(
        GPPrior(deepcopy(kernel), deepcopy(mean)),
        Posterior(dim, zeros(T, dim), cholesky(Matrix{T}(I(dim)))),
        deepcopy(opt),
    )
end

@traitimpl IsFull{LatentGP}

## AbstractVarLatent

abstract type AbstractVarLatent{T,Tpr,Tpo} <: AbstractLatent{T,Tpr,Tpo} end

## Variational Gaussian Process
mutable struct VarLatent{T,Tpr<:GPPrior,Tpo<:VarPosterior{T},O} <:
               AbstractVarLatent{T,Tpr,Tpo}
    prior::Tpr
    post::Tpo
    opt::O
end

function VarLatent(T::DataType, dim::Int, kernel::Kernel, mean::PriorMean, opt)
    return VarLatent(
        GPPrior(deepcopy(kernel), deepcopy(mean)), VarPosterior{T}(dim), deepcopy(opt)
    )
end

@traitimpl IsFull{VarLatent}

## Sparse Variational Gaussian Process

mutable struct SparseVarLatent{
    T,Tpr<:GPPrior,Tpo<:VarPosterior{T},Topt,TZ<:AbstractVector,TZopt
} <: AbstractVarLatent{T,Tpr,Tpo}
    prior::Tpr
    post::Tpo
    Z::TZ
    opt::Topt
    Zopt::TZopt
end

function SparseVarLatent(
    T::DataType,
    Z::AbstractVector,
    kernel::Kernel,
    mean::PriorMean,
    opt=nothing,
    Zopt=nothing,
)
    dim = length(Z)
    return SparseVarLatent(
        GPPrior(deepcopy(kernel), deepcopy(mean)),
        VarPosterior{T}(dim),
        deepcopy(Z),
        deepcopy(opt),
        deepcopy(Zopt),
    )
end

@traitimpl IsSparse{SparseVarLatent}

## Monte-Carlo Gaussian Process

struct SampledLatent{T,Tpr<:GPPrior,Tpo<:SampledPosterior{T}} <: AbstractLatent{T,Tpr,Tpo}
    prior::Tpr
    post::Tpo
end

function SampledLatent(T::DataType, dim::Int, kernel::Kernel, mean::PriorMean)
    return SampledLatent(
        GPPrior(deepcopy(kernel), deepcopy(mean)),
        SampledPosterior(dim, zeros(T, dim), Symmetric(Matrix{T}(I(dim)))),
    )
end

@traitimpl IsFull{SampledLatent}

## Online Sparse Variational Process

mutable struct OnlineVarLatent{
    T,
    Tpr<:GPPrior,
    Tpo<:AbstractVarPosterior{T},
    Topt,
    TZ<:AbstractVector,
    TZalg<:InducingPoints.OnIPSA,
    TZopt,
} <: AbstractVarLatent{T,Tpo,Tpr}
    prior::Tpr
    post::Tpo
    Z::TZ
    Zₐ::TZ
    Zalg::TZalg
    Zupdated::Bool
    opt::Topt
    Zopt::TZopt
end

function OnlineVarLatent(
    T::DataType,
    dim::Int,
    Z::AbstractVector,
    Zalg::InducingPoints.OnIPSA,
    kernel::Kernel,
    mean::PriorMean,
    opt=nothing,
    Zopt=nothing,
)
    return OnlineVarLatent(
        GPPrior(deepcopy(kernel), deepcopy(mean)),
        OnlineVarPosterior{T}(dim),
        Z,
        deepcopy(Z),
        Zalg,
        false,
        deepcopy(opt),
        deepcopy(Zopt),
    )
end

@traitimpl IsSparse{OnlineVarLatent}

## Variational Student-T Process

mutable struct TVarLatent{T<:Real,Tpr<:TPrior,Tpo<:VarPosterior{T},O} <:
               AbstractVarLatent{T,Tpr,Tpo}
    prior::Tpr
    post::Tpo
    opt::O
end

function TVarLatent(T::DataType, ν::Real, dim::Int, kernel::Kernel, mean::PriorMean, opt)
    return TVarLatent(
        TPrior(deepcopy(kernel), deepcopy(mean), ν, rand(T), rand(T)),
        VarPosterior{T}(dim),
        deepcopy(opt),
    )
end

@traitimpl IsFull{TVarLatent}

### Functions

prior(gp::AbstractLatent) = gp.prior
kernel(gp::AbstractLatent) = kernel(prior(gp))
setkernel!(gp::AbstractLatent, kernel::Kernel) = setkernel!(prior(gp), kernel)
pr_mean(gp::AbstractLatent) = mean(prior(gp))
pr_mean(gp::AbstractLatent, X::AbstractVector) = mean(prior(gp), X)
setpr_mean!(gp::AbstractLatent, μ₀::PriorMean) = setmean!(prior(gp), μ₀)
pr_cov(gp::AbstractLatent) = cov(prior(gp))
pr_cov(gp::TVarLatent) = prior(gp).χ * cov(prior(gp))
pr_cov!(gp::AbstractLatent, K::Cholesky) = gp.prior.K = K

posterior(gp::AbstractLatent) = gp.post
Distributions.dim(gp::AbstractLatent) = dim(posterior(gp))
Distributions.mean(gp::AbstractLatent) = mean(posterior(gp))
Distributions.cov(gp::AbstractLatent) = cov(posterior(gp))
Distributions.var(gp::AbstractLatent) = var(posterior(gp))
nat1(gp::AbstractVarLatent) = nat1(posterior(gp))
nat2(gp::AbstractVarLatent) = nat2(posterior(gp))

mean_f(model::AbstractGPModel, kernel_matrices) = mean_f.(model.f, kernel_matrices)

@traitfn function mean_f(gp::T, kernel_matrices) where {T <: AbstractLatent; IsFull{T}}
    return mean_f(mean(gp))
end
@traitfn function mean_f(gp::T, kernel_matrices) where {T <: AbstractLatent; !IsFull{T}}
    return mean_f(mean(gp), kernel_matrices.κ)
end

mean_f(μ::AbstractVector) = μ
mean_f(μ::AbstractVector, κ::AbstractMatrix) = κ * μ

var_f(model::AbstractGPModel, kernel_matrices) = var_f.(model.f, kernel_matrices)

@traitfn function var_f(gp::T, kernel_matrices) where {T <: AbstractLatent; IsFull{T}}
    return var_f(cov(gp))
end
@traitfn function var_f(gp::T, kernel_matrices) where {T <: AbstractLatent; !IsFull{T}}
    return var_f(cov(gp), kernel_matrices.κ, kernel_matrices.K̃)
end

var_f(Σ::AbstractMatrix) = diag(Σ)
var_f(Σ::AbstractMatrix, κ::AbstractMatrix, K̃::AbstractVector) = diag_ABt(κ * Σ, κ) + K̃

Zview(gp::SparseVarLatent) = gp.Z
Zview(gp::OnlineVarLatent) = gp.Z

setZ!(gp::AbstractLatent, Z::AbstractVector) = gp.Z = Z

opt(gp::AbstractLatent) = gp.opt
Zopt(::AbstractLatent) = nothing
Zopt(gp::SparseVarLatent) = gp.Zopt
Zopt(gp::OnlineVarLatent) = gp.Zopt

function compute_K(gp::AbstractLatent, X::AbstractVector, jitt::Real)
    return cholesky(kernelmatrix(kernel(gp), X) + jitt * I)
end

function compute_K(gp::AbstractLatent, jitt::Real)
    return cholesky(kernelmatrix(kernel(gp), gp.Z) + jitt * I)
end

function compute_κ(gp::SparseVarLatent, X::AbstractVector, K, jitt::Real)
    Knm = kernelmatrix(kernel(gp), X, gp.Z)
    κ = copy(Knm / K)
    K̃ = kernelmatrix_diag(kernel(gp), X) .+ jitt - diag_ABt(κ, Knm)
    all(K̃ .> 0) || error("K̃ has negative values")
    return (; Knm, κ, K̃)
end

function compute_κ(gp::OnlineVarLatent, X::AbstractVector, K, jitt::Real)
    # Covariance with the model at t-1
    Kab = kernelmatrix(kernel(gp), gp.Zₐ, gp.Z)
    κₐ = Kab / K
    Kₐ = Symmetric(kernelmatrix(kernel(gp), gp.Zₐ) + jitt * I)
    K̃ₐ = Kₐ - κₐ * transpose(Kab)

    # Covariance with a new batch
    Knm = kernelmatrix(kernel(gp), X, gp.Z)
    κ = Knm / K
    K̃ = kernelmatrix_diag(kernel(gp), X) .+ jitt - diag_ABt(κ, Knm)
    all(K̃ .> 0) || error("K̃ has negative values")
    return (; Kab, κₐ, K̃ₐ, Knm, κ, K̃)
end
