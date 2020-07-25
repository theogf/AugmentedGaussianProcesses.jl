abstract type MaternLikelihood{T<:Real} <: RegressionLikelihood{T} end


"""
**Matern 3/2 likelihood**

Matern 3/2 likelihood for regression: ````
see [wiki page](https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function)

```julia
Matern3_2Likelihood(ρ::T) #ρ is the lengthscale
```

---

For the analytical solution, it is augmented via:
```math
p(y|f,\\omega) = \\mathcal{N}(y|f,\\sigma^2\\omega)
```
Where ``\\omega \\sim \\mathcal{IG}(\\frac{\\nu}{2},\\frac{\\nu}{2})`` where ``\\mathcal{IG}`` is the inverse gamma distribution
See paper [Robust Gaussian Process Regression with a Student-t Likelihood](http://www.jmlr.org/papers/volume12/jylanki11a/jylanki11a.pdf)
"""
struct Matern3_2Likelihood{T<:Real,A<:AbstractVector{T}} <: MaternLikelihood{T}
    ρ::T
    c::A
    θ::A
    function Matern3_2Likelihood{T}(ρ::T) where {T<:Real}
        new{T,Vector{T}}(ρ)
    end
    function Matern3_2Likelihood{T}(
        ρ::T,
        c²::A,
        θ::A,
    ) where {T<:Real, A<:AbstractVector{T}}
        new{T, A}(ρ, c², θ)
    end
end

function Matern3_2Likelihood(ρ::T = 1.0) where {T<:Real}
    Matern3_2Likelihood{T}(ρ)
end

implemented(
    ::Matern3_2Likelihood,
    ::Union{<:AnalyticVI,<:QuadratureVI,<:GibbsSampling},
) = true

function init_likelihood(
    likelihood::Matern3_2Likelihood{T},
    inference::Inference{T},
    nLatent::Int,
    nSamplesUsed::Int,
    nFeatures::Int,
) where {T}
    if inference isa AnalyticVI || inference isa GibbsSampling
        Matern3_2Likelihood{T}(
            likelihood.ρ,
            abs2.(T.(rand(T, nSamplesUsed))),
            zeros(T, nSamplesUsed),
        )
    else
        Matern3_2Likelihood{T}(likelihood.ρ)
    end
end

function pdf(l::Matern3_2Likelihood{T}, y::Real, f::Real) where {T}
    u = sqrt(3) * abs(y - f) / l.ρ
    4 * l.ρ / sqrt(3) * (one(T) + u) * exp(-u)
end

function logpdf(l::Matern3_2Likelihood{T}, y::Real, f::Real) where {T}
    u = sqrt(3) * abs(y - f) / l.ρ
    log(4 * l.ρ / sqrt(3)) + log(one(T) + u) - u
end

function Base.show(io::IO, model::Matern3_2Likelihood{T}) where {T}
    print(io, "Matern 3/2 likelihood")
end

function compute_proba(
    l::Matern3_2Likelihood{T},
    μ::AbstractVector{<:Real},
    σ²::AbstractVector{<:Real},
) where {T<:Real}
    return μ, max.(σ², 0.0) .+ 4 * l.ρ^2 / 3
end

## Local Updates ##
function local_updates!(
    l::Matern3_2Likelihood,
    y::AbstractVector,
    μ::AbstractVector{T},
    diag_cov::AbstractVector{T},
) where {T<:Real}
    l.c .= sqrt.(diag_cov + abs2.(μ - y))
    l.θ .= 3.0 ./ (2.0 .* sqrt.(3) * l.c * l.ρ .+ 2 * l.ρ^2)
end

function sample_local!(
    l::Matern3_2Likelihood,
    y::AbstractVector,
    f::AbstractVector,
) where {T}
    l.c .=
        rand.(GeneralizedInverseGaussian.(
            3 / (2 * l.ρ^2),
            2.0 .* abs2.(f - y),
            1.5,
        ))
    set_ω!(l, l.c)
    return nothing
end

@inline ∇E_μ(
    l::Matern3_2Likelihood,
    ::AOptimizer,
    y::AbstractVector,
) where {T} = (2.0 * l.θ .* y,)
@inline ∇E_Σ(
    l::Matern3_2Likelihood,
    ::AOptimizer,
    y::AbstractVector,
) where {T} = (l.θ,)

## ELBO  ##

function expecLogLikelihood(
    l::Matern3_2Likelihood,
    i::AnalyticVI,
    y::AbstractVector,
    μ::AbstractVector,
    diag_cov::AbstractVector,
) where {T}
    throw(error("Not implemented yet"))
end

function InverseGammaKL(l::Matern3_2Likelihood{T}) where {T}
    α_p = model.likelihood.ν / 2
    β_p = α_p * model.likelihood.σ^2
    model.inference.ρ * sum(broadcast(
        InverseGammaKL,
        model.likelihood.α,
        model.likelihood.β,
        α_p,
        β_p,
    ))
end

## PDF and Log PDF Gradients ## (verified gradients)

@inline function grad_logpdf(
    l::Matern3_2Likelihood{T},
    y::Real,
    f::Real,
) where {T<:Real}
    3.0 * (y - f) / (l.ρ * (abs(f - y) * sqrt(3) + l.ρ))
end

@inline function hessian_logpdf(
    l::Matern3_2Likelihood{T},
    y::Real,
    f::Real,
) where {T<:Real}
    3.0 / (l.ρ + sqrt(3) * abs(f - y))^2
end
