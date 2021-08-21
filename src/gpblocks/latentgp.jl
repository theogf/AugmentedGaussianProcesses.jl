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
        GPPrior(deepcopy(kernel), deepcopy(mean), cholesky(Matrix{T}(I, dim, dim))),
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
        GPPrior(deepcopy(kernel), deepcopy(mean), cholesky(Matrix{T}(I, dim, dim))),
        VarPosterior{T}(dim),
        deepcopy(opt),
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
    Knm::Matrix{T}
    κ::Matrix{T}
    K̃::Vector{T}
    opt::Topt
    Zopt::TZopt
end

function SparseVarLatent(
    T::DataType,
    dim::Int,
    S::Int,
    Z::AbstractVector,
    kernel::Kernel,
    mean::PriorMean,
    opt=nothing,
    Zopt=nothing
)
    return SparseVarLatent(
        GPPrior(deepcopy(kernel), deepcopy(mean), cholesky(Matrix{T}(I(dim)))),
        VarPosterior{T}(dim),
        deepcopy(Z),
        Matrix{T}(undef, S, dim),
        Matrix{T}(undef, S, dim),
        Vector{T}(undef, S),
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
        GPPrior(deepcopy(kernel), deepcopy(mean), cholesky(Matrix{T}(I, dim, dim))),
        SampledPosterior(dim, zeros(T, dim), Symmetric(Matrix{T}(I(dim)))),
    )
end

@traitimpl IsFull{SampledLatent}

## Online Sparse Variational Process

mutable struct OnlineVarLatent{T,Tpr<:GPPrior,Tpo<:AbstractVarPosterior{T},Topt,TZ<:AbstractVector,
    TZalg<:InducingPoints.OnIPSA,TZopt} <:
               AbstractVarLatent{T,Tpo,Tpr}
    prior::Tpr
    post::Tpo
    Z::TZ
    Zalg::TZalg
    Knm::Matrix{T}
    κ::Matrix{T}
    K̃::Vector{T}
    Zupdated::Bool
    opt::Topt
    Zopt::TZopt
    Zₐ::AbstractVector
    Kab::Matrix{T}
    κₐ::Matrix{T}
    K̃ₐ::Matrix{T}
    invDₐ::Symmetric{T,Matrix{T}}
    prev𝓛ₐ::T
    prevη₁::Vector{T}
end

function OnlineVarLatent(
    T::DataType,
    dim::Int,
    nSamplesUsed::Int,
    Z::AbstractVector,
    Zalg::InducingPoints.OnIPSA,
    kernel::Kernel,
    mean::PriorMean,
    opt=nothing,
    Zopt=nothing
)
    return OnlineVarLatent(
        GPPrior(deepcopy(kernel), deepcopy(mean), cholesky(Matrix{T}(I, dim, dim))),
        OnlineVarPosterior{T}(dim),
        Z,
        Zalg,
        Matrix{T}(undef, nSamplesUsed, dim),
        Matrix{T}(undef, nSamplesUsed, dim),
        Vector{T}(undef, nSamplesUsed),
        false,
        deepcopy(opt),
        deepcopy(Zopt),
        vec(Z),
        Matrix{T}(I, dim, dim),
        Matrix{T}(I, dim, dim),
        Matrix{T}(I, dim, dim),
        Symmetric(Matrix{T}(I, dim, dim)),
        zero(T),
        Vector{T}(undef, dim),
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
        TPrior(
            deepcopy(kernel),
            deepcopy(mean),
            cholesky(Matrix{T}(I, dim, dim)),
            ν,
            rand(T),
            rand(T),
        ),
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

mean_f(model::AbstractGP) = mean_f.(model.f)

@traitfn mean_f(gp::T) where {T <: AbstractLatent; IsFull{T}} = mean_f(mean(gp))
@traitfn mean_f(gp::T) where {T <: AbstractLatent; !IsFull{T}} = mean_f(mean(gp), gp.κ)

mean_f(μ::AbstractVector) = μ
mean_f(μ::AbstractVector, κ::AbstractMatrix) = κ * μ

var_f(model::AbstractGP) = var_f.(model.f)

@traitfn var_f(gp::T) where {T <: AbstractLatent; IsFull{T}} = var_f(cov(gp))
@traitfn var_f(gp::T) where {T <: AbstractLatent; !IsFull{T}} = var_f(cov(gp), gp.κ, gp.K̃)

var_f(Σ::AbstractMatrix) = diag(Σ)
var_f(Σ::AbstractMatrix, κ::AbstractMatrix, K̃::AbstractVector) = diag_ABt(κ * Σ, κ) + K̃

Zview(gp::SparseVarLatent) = gp.Z
Zview(gp::OnlineVarLatent) = gp.Z

setZ!(gp::AbstractLatent, Z::AbstractVector) = gp.Z = Z

opt(gp::AbstractLatent) = gp.opt
Zopt(::AbstractLatent) = nothing
Zopt(gp::SparseVarLatent) = gp.Zopt
Zopt(gp::OnlineVarLatent) = gp.Zopt

@traitfn function compute_K!(
    gp::TGP, X::AbstractVector, jitt::Real
) where {TGP <: AbstractLatent; IsFull{TGP}}
    return pr_cov!(gp, cholesky(kernelmatrix(kernel(gp), X) + jitt * I))
end

@traitfn function compute_K!(gp::T, jitt::Real) where {T <: AbstractLatent; !IsFull{T}}
    return pr_cov!(gp, cholesky(kernelmatrix(kernel(gp), gp.Z) + jitt * I))
end

function compute_κ!(gp::SparseVarLatent, X::AbstractVector, jitt::Real)
    gp.Knm = kernelmatrix(kernel(gp), X, gp.Z)
    gp.κ = copy(gp.Knm / pr_cov(gp))
    gp.K̃ = kernelmatrix_diag(kernel(gp), X) .+ jitt - diag_ABt(gp.κ, gp.Knm)
    return all(gp.K̃ .> 0) || error("K̃ has negative values")
end

function compute_κ!(gp::OnlineVarLatent, X::AbstractVector, jitt::Real)
    # Covariance with the model at t-1
    gp.Kab = kernelmatrix(kernel(gp), gp.Zₐ, gp.Z)
    gp.κₐ = gp.Kab / pr_cov(gp)
    Kₐ = Symmetric(kernelmatrix(kernel(gp), gp.Zₐ) + jitt * I)
    gp.K̃ₐ = Kₐ - gp.κₐ * transpose(gp.Kab)

    # Covariance with a new batch
    gp.Knm = kernelmatrix(kernel(gp), X, gp.Z)
    gp.κ = gp.Knm / pr_cov(gp)
    gp.K̃ = kernelmatrix_diag(kernel(gp), X) .+ jitt - diag_ABt(gp.κ, gp.Knm)
    @assert all(gp.K̃ .> 0) "K̃ has negative values"
end
