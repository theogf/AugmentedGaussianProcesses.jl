## Gaussian Process
abstract type AbstractGPPrior{T<:Real,K<:Kernel,Tmean<:PriorMean} end

kernel(gp::AbstractGPPrior) = gp.kernel
mean(gp::AbstractGPPrior) = gp.μ₀
mean(gp::AbstractGPPrior, X::AbstractVector) = gp.μ₀(X)
cov(gp::AbstractGPPrior) = gp.K

mutable struct GPPrior{T,K<:Kernel,Tmean<:PriorMean} <: AbstractGPPrior{T,K,Tmean}
    kernel::K
    μ₀::Tmean
    K::PDMat{T,Matrix{T}}
end

mutable struct TPrior{T,K<:Kernel,Tmean<:PriorMean} <: AbstractGPPrior{T,K,Tmean}
    kernel::K
    μ₀::Tmean
    K::PDMat{T,Matrix{T}}
    ν::T # Number of degrees of freedom
    l²::T # Expectation of ||L^{-1}(f-μ⁰)||₂²
    χ::T  # Expectation of σ
end

abstract type AbstractPosterior{T<:Real} end

dim(p::AbstractPosterior) = p.dim
mean(p::AbstractPosterior) = p.μ
cov(p::AbstractPosterior) = p.Σ
var(p::AbstractPosterior) = diag(p.Σ)

mutable struct Posterior{T<:Real} <: AbstractPosterior{T}
    dim::Int
    α::Vector{T} # Σ⁻¹ (y - μ₀)
    Σ::PDMat{T,Matrix{T}} # Posterior Covariance : K + σ²I
end

mean(p::Posterior) = p.α

abstract type AbstractVarPosterior{T} <: AbstractPosterior{T} end

nat1(p::AbstractVarPosterior) = p.η₁
nat2(p::AbstractVarPosterior) = p.η₂

struct VarPosterior{T} <: AbstractVarPosterior{T}
    dim::Int
    μ::Vector{T}
    Σ::Symmetric{T,Matrix{T}}
    η₁::Vector{T}
    η₂::Symmetric{T,Matrix{T}}
end

VarPosterior{T}(dim::Int) where {T<:Real} = VarPosterior{T}(
    dim,
    zeros(T, dim),
    Symmetric(Matrix{T}(I, dim, dim)),
    zeros(T, dim),
    Symmetric(Matrix{T}(-0.5 * I, dim, dim)),
)

mutable struct OnlineVarPosterior{T} <: AbstractVarPosterior{T}
    dim::Int
    μ::Vector{T}
    Σ::Symmetric{T,Matrix{T}}
    η₁::Vector{T}
    η₂::Symmetric{T,Matrix{T}}
end

OnlineVarPosterior{T}(dim::Int) where {T<:Real} = OnlineVarPosterior{T}(
    dim,
    zeros(T, dim),
    Symmetric(Matrix{T}(I, dim, dim)),
    zeros(T, dim),
    Symmetric(Matrix{T}(-0.5 * I, dim, dim)),
)

struct SampledPosterior{T} <: AbstractPosterior{T}
    dim::Int
    f::Vector{T}
    Σ::Symmetric{T, Matrix{T}}
end

mean(p::SampledPosterior) = p.f

#### Latent models ####

## Exact Gaussian Process
struct LatentGP{T,Tpr<:GPPrior,Tpo<:Posterior{T},O} <: AbstractLatent{T,Tpr,Tpo}
    prior::Tpr
    post::Tpo
    opt::O
end

function LatentGP(
    T::DataType,
    dim::Int,
    kernel::Kernel,
    mean::PriorMean,
    opt,
)
    LatentGP(
        GPPrior(
            deepcopy(kernel),
            deepcopy(mean),
            PDMat(Matrix{T}(I, dim, dim)),
        ),
        Posterior(dim, zeros(T, dim), PDMat(Matrix{T}(I(dim)))),
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

function VarLatent(
    T::DataType,
    dim::Int,
    kernel::Kernel,
    mean::PriorMean,
    opt,
)
    VarLatent(
        GPPrior(
            deepcopy(kernel),
            deepcopy(mean),
            PDMat(Matrix{T}(I, dim, dim)),
        ),
        VarPosterior{T}(dim),
        deepcopy(opt),
    )
end

@traitimpl IsFull{VarLatent}

## Sparse Variational Gaussian Process

struct SparseVarLatent{
    T,
    Tpr<:GPPrior,
    Tpo<:VarPosterior{T},
    TZ<:AbstractInducingPoints,
    O,
} <: AbstractVarLatent{T,Tpr,Tpo}
    prior::Tpr
    post::Tpo
    Z::TZ
    Knm::Matrix{T}
    κ::Matrix{T}
    K̃::Vector{T}
    opt::O
end

function SparseVarLatent(
    T::DataType,
    dim::Int,
    S::Int,
    Z::AbstractInducingPoints,
    kernel::Kernel,
    mean::PriorMean,
    opt,
)
    SparseVarLatent(
        GPPrior(
            deepcopy(kernel),
            deepcopy(mean),
            PDMat(Matrix{T}(I(dim))),
        ),
        VarPosterior{T}(dim),
        deepcopy(Z),
        Matrix{T}(undef, S, dim),
        Matrix{T}(undef, S, dim),
        Vector{T}(undef, S),
        deepcopy(opt),
    )
end

@traitimpl IsSparse{SparseVarLatent}

## Monte-Carlo Gaussian Process

struct SampledLatent{T,Tpr<:GPPrior,Tpo<:SampledPosterior{T}} <:
               AbstractLatent{T,Tpr,Tpo}
    prior::Tpr
    post::Tpo
end

function SampledLatent(
    T::DataType,
    dim::Int,
    kernel::Kernel,
    mean::PriorMean,
)
    SampledLatent(
        GPPrior(
            deepcopy(kernel),
            deepcopy(mean),
            PDMat(Matrix{T}(I, dim, dim)),
        ),
        SampledPosterior(dim, zeros(T, dim), Symmetric(Matrix{T}(I(dim)))),
    )
end

@traitimpl IsFull{SampledLatent}

## Online Sparse Variational Process

mutable struct OnlineVarLatent{
    T,
    Tpr<:GPPrior,
    Tpo<:VarPosterior{T},
    O,
} <: AbstractVarLatent{T,Tpo,Tpr}
    prior::Tpr
    post::Tpo
    Z::InducingPoints.AIP
    Knm::Matrix{T}
    κ::Matrix{T}
    K̃::Vector{T}
    Zupdated::Bool
    opt::O
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
    Z::AbstractInducingPoints,
    kernel::Kernel,
    mean::PriorMean,
    opt,
)
    OnlineVarLatent(
        GPPrior(
            deepcopy(kernel),
            deepcopy(mean),
            PDMat(Matrix{T}(I, dim, dim)),
        ),
        OnlineVarPosterior{T}(dim),
        Z,
        Matrix{T}(undef, nSamplesUsed, dim),
        Matrix{T}(undef, nSamplesUsed, dim),
        Vector{T}(undef, nSamplesUsed),
        false,
        deepcopy(opt),
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
               AbstractLatent{T,Tpr,Tpo}
    prior::Tpr
    post::Tpo
    opt::O
end

function TVarLatent(
    T::DataType,
    ν::Real,
    dim::Int,
    kernel::Kernel,
    mean::PriorMean,
    opt,
)
    TVarLatent(
        TPrior(
            deepcopy(kernel),
            deepcopy(mean),
            PDMat(Matrix{T}(I, dim, dim)),
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
pr_mean(gp::AbstractLatent) = mean(prior(gp))
pr_mean(gp::AbstractLatent, X::AbstractVector) = mean(prior(gp), X)
pr_cov(gp::AbstractLatent) = cov(prior(gp))
pr_cov(gp::TVarLatent) = prior(gp).χ * cov(prior(gp))
pr_cov!(gp::AbstractLatent, K::PDMat) = gp.prior.K = K

posterior(gp::AbstractLatent) = gp.post
dim(gp::AbstractLatent) = dim(posterior(gp))
mean(gp::AbstractLatent) = mean(posterior(gp))
cov(gp::AbstractLatent) = cov(posterior(gp))
var(gp::AbstractLatent) = var(posterior(gp))
nat1(gp::AbstractVarLatent) = nat1(posterior(gp))
nat2(gp::AbstractVarLatent) = nat2(posterior(gp))

mean_f(model::AbstractGP) = mean_f.(model.f)

@traitfn mean_f(gp::T) where {T <: AbstractLatent; IsFull{T}} = mean(gp)
@traitfn mean_f(gp::T) where {T <: AbstractLatent; !IsFull{T}} = gp.κ * mean(gp)

var_f(model::AbstractGP) = var_f.(model.f)

@traitfn var_f(gp::T) where {T <: AbstractLatent; IsFull{T}} = var(gp)
@traitfn var_f(gp::T) where {T <: AbstractLatent; !IsFull{T}} = opt_diag(gp.κ * cov(gp), gp.κ) + gp.K̃

Zview(gp::SparseVarLatent) = gp.Z
Zview(gp::OnlineVarLatent) = gp.Z

@traitfn compute_K!(
    gp::TGP,
    X::AbstractVector,
    jitt::Real,
) where {TGP <: AbstractLatent; IsFull{TGP}} =
    pr_cov!(gp, PDMat(kernelmatrix(kernel(gp), X) + jitt * I))

@traitfn compute_K!(gp::T, jitt::Real) where {T <: AbstractLatent; !IsFull{T}} =
    pr_cov!(gp, PDMat(kernelmatrix(kernel(gp), gp.Z) + jitt * I))

function compute_κ!(gp::SparseVarLatent, X::AbstractVector, jitt::Real)
    gp.Knm .= kernelmatrix(kernel(gp), X, gp.Z)
    gp.κ .= gp.Knm / pr_cov(gp)
    gp.K̃ .=
        kerneldiagmatrix(kernel(gp), X) .+ jitt -
        opt_diag(gp.κ, gp.Knm)

    @assert all(gp.K̃ .> 0) "K̃ has negative values"
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
    gp.K̃ = kerneldiagmatrix(kernel(gp), X) .+ jitt - opt_diag(gp.κ, gp.Knm)
    @assert all(gp.K̃ .> 0) "K̃ has negative values"
end
