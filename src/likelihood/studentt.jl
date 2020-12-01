"""
```julia
StudentTLikelihood(ν::T,σ::Real=one(T))
```
[Student-t likelihood](https://en.wikipedia.org/wiki/Student%27s_t-distribution) for regression:
```math
    p(y|f,ν,σ) = Γ(0.5(ν+1))/(sqrt(νπ) σ Γ(0.5ν)) * (1+(y-f)^2/(σ^2ν))^(-0.5(ν+1))
```
`ν` is the number of degrees of freedom and `σ` is the variance for local scale of the data.

---

For the analytical solution, it is augmented via:
```math
    p(y|f,ω) = N(y|f,σ^2 ω)
```
Where `ω ~ IG(0.5ν,,0.5ν)` where `IG` is the inverse gamma distribution
See paper [Robust Gaussian Process Regression with a Student-t Likelihood](http://www.jmlr.org/papers/volume12/jylanki11a/jylanki11a.pdf)
"""
mutable struct StudentTLikelihood{T<:Real,A<:AbstractVector{T}} <:
               RegressionLikelihood{T}
    ν::T
    α::T
    σ::T
    c::A
    θ::A
    function StudentTLikelihood{T}(ν::T, σ::T = one(T)) where {T<:Real}
        new{T,Vector{T}}(ν, (ν + one(T)) / 2.0, σ)
    end
    function StudentTLikelihood{T}(
        ν::T,
        σ::T,
        c::A,
        θ::A,
    ) where {T<:Real,A<:AbstractVector{T}}
        new{T,A}(ν, (ν + one(T)) / 2.0, σ, c, θ)
    end
end

function StudentTLikelihood(ν::T, σ::T = one(T)) where {T<:Real}
    StudentTLikelihood{T}(ν, σ)
end

implemented(
    ::StudentTLikelihood,
    ::Union{<:AnalyticVI,<:QuadratureVI,<:GibbsSampling},
) = true

function init_likelihood(
    likelihood::StudentTLikelihood{T},
    inference::Inference{T},
    ::Int,
    nSamplesUsed::Int,
) where {T}
    if inference isa AnalyticVI || inference isa GibbsSampling
        StudentTLikelihood{T}(
            likelihood.ν,
            likelihood.σ,
            rand(T, nSamplesUsed),
            zeros(T, nSamplesUsed),
        )
    else
        StudentTLikelihood{T}(likelihood.ν, likelihood.σ)
    end
end

function (l::StudentTLikelihood)(y::Real, f::Real)
    tdistpdf(l.ν, (y - f) / l.σ)
end

function Distributions.loglikelihood(l::StudentTLikelihood, y::Real, f::Real)
    tdistlogpdf(l.ν, (y - f) / l.σ)
end

function Base.show(io::IO, model::StudentTLikelihood{T}) where {T}
    print(io, "Student-t likelihood")
end

function compute_proba(
    l::StudentTLikelihood{T},
    μ::AbstractVector{<:Real},
    σ²::AbstractVector{<:Real},
) where {T<:Real}
    return μ, max.(σ², zero(σ²)) .+ 0.5 * l.ν * l.σ^2 / (0.5 * l.ν - 1)
end

## Local Updates ##

function local_updates!(
    l::StudentTLikelihood{T},
    y::AbstractVector,
    μ::AbstractVector,
    diag_cov::AbstractVector,
) where {T}
    l.c .= 0.5 * (diag_cov + abs2.(μ - y) .+ l.σ^2 * l.ν)
    l.θ = l.α ./ l.c
end

function sample_local!(
    l::StudentTLikelihood{T},
    y::AbstractVector,
    f::AbstractVector,
) where {T}
    l.c .= rand.(InverseGamma.(l.α, 0.5 * (abs2.(f - y) .+ l.σ^2 * l.ν)))
    set_ω!(l, inv.(l.c))
    return nothing
end

## Global Gradients ##

@inline ∇E_μ(l::StudentTLikelihood, ::AOptimizer, y::AbstractVector) =
    (l.θ .* y,)
@inline ∇E_Σ(l::StudentTLikelihood, ::AOptimizer, ::AbstractVector) =
    (0.5 .* l.θ,)

## ELBO Section ##

function expec_log_likelihood(
    l::StudentTLikelihood{T},
    ::AnalyticVI,
    y::AbstractVector,
    μ::AbstractVector,
    diag_cov::AbstractVector,
) where {T}
    tot = -0.5 * length(y) * (log(twoπ * l.σ^2))
    tot += -sum(log.(l.c) .- digamma(l.α))
    tot +=
        -0.5 * (
            dot(l.θ, diag_cov) + dot(l.θ, abs2.(μ)) - 2.0 * dot(l.θ, μ .* y) +
            dot(l.θ, abs2.(y))
        )
    return tot
end

AugmentedKL(l::StudentTLikelihood, ::AbstractVector) = InverseGammaKL(l)

function InverseGammaKL(l::StudentTLikelihood{T}) where {T}
    α_p = l.ν / 2
    β_p = α_p * l.σ^2
    InverseGammaKL(l.α, l.c, α_p, β_p)
end

## PDF and Log PDF Gradients ## (verified gradients)

function grad_loglike(l::StudentTLikelihood{T}, y::Real, f::Real) where {T<:Real}
    (one(T) + l.ν) * (y - f) / ((f - y)^2 + l.σ^2 * l.ν)
end

function hessian_loglike(
    l::StudentTLikelihood{T},
    y::Real,
    f::Real,
) where {T<:Real}
    v = l.ν * l.σ^2
    Δ² = (f - y)^2
    (one(T) + l.ν) * (-v + Δ²) / (v + Δ²)^2
end
