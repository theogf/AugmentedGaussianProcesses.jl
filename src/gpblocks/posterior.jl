abstract type AbstractPosterior{T<:Real} end

Distributions.dim(p::AbstractPosterior) = p.dim
Distributions.mean(p::AbstractPosterior) = p.μ
Distributions.cov(p::AbstractPosterior) = p.Σ
Distributions.var(p::AbstractPosterior) = diag(p.Σ)

mutable struct Posterior{T<:Real} <: AbstractPosterior{T}
    dim::Int
    α::Vector{T} # Σ⁻¹ (y - μ₀)
    Σ::PDMat{T,Matrix{T}} # Posterior Covariance : K + σ²I
end

Distributions.mean(p::Posterior) = p.α

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

Distributions.mean(p::SampledPosterior) = p.f
