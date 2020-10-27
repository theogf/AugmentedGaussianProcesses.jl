"""
```julia
LaplaceLikelihood(β::T=1.0)  #  Laplace likelihood with scale β
```

Laplace likelihood for regression:
```math
1/(2β) exp(-|y-f|/β)
```
see [wiki page](https://en.wikipedia.org/wiki/Laplace_distribution)
---
For the analytical solution, it is augmented via:
```math
p(y|f,ω) = N(y|f,ω⁻¹)
```
where ``ω ~ Exp(ω | 1/(2 β^2))``, and `Exp` is the [Exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution)
We use the variational distribution ``q(ω) = GIG(ω | a,b,p)``
"""
mutable struct LaplaceLikelihood{T<:Real, A<:AbstractVector{T}} <: RegressionLikelihood{T}
    β::T
    a::T
    p::T
    b::A #Variational parameter b of GIG
    θ::A #Expected value of ω
    function LaplaceLikelihood{T}(β::T) where {T<:Real}
        new{T, Vector{T}}(β, β^-2, 0.5)
    end
    function LaplaceLikelihood{T}(
        β::T,
        b::A,
        θ::AbstractVector{T},
    ) where {T<:Real, A<:AbstractVector{T}}
        new{T, A}(β, β^(-2), 0.5, b, θ)
    end
end

function LaplaceLikelihood(β::T = 1.0) where {T<:Real}
    LaplaceLikelihood{T}(β)
end

implemented(
    ::LaplaceLikelihood,
    ::Union{<:AnalyticVI,<:QuadratureVI,<:GibbsSampling},
) = true

function init_likelihood(
    likelihood::LaplaceLikelihood{T},
    inference::Inference{T},
    nLatent::Int,
    nSamplesUsed::Int,
) where {T}
    if inference isa AnalyticVI || inference isa GibbsSampling
        LaplaceLikelihood{T}(
            likelihood.β,
            rand(T, nSamplesUsed),
            zeros(T, nSamplesUsed),
        )
    else
        LaplaceLikelihood{T}(likelihood.β)
    end
end

function pdf(l::LaplaceLikelihood, y::Real, f::Real)
    Distributions.pdf(Laplace(f, l.β), y)
end

function logpdf(l::LaplaceLikelihood, y::Real, f::Real)
    Distributions.logpdf(Laplace(f, l.β), y)
end

function Base.show(io::IO, model::LaplaceLikelihood{T}) where {T}
    print(io, "Laplace likelihood")
end

function compute_proba(
    l::LaplaceLikelihood{T},
    μ::AbstractVector{<:Real},
    σ²::AbstractVector{<:Real},
) where {T<:Real}
    return μ, max.(σ², 0.0) .+ 2 * l.β^2
end

## Local Updates ##

function local_updates!(
    l::LaplaceLikelihood{T},
    y::AbstractVector,
    μ::AbstractVector,
    diagΣ::AbstractVector,
) where {T}
    @. l.b = diagΣ + abs2(μ - y)
    @. l.θ = sqrt(l.a) / sqrt.(l.b)
end

function sample_local!(
    l::LaplaceLikelihood,
    y::AbstractVector,
    f::AbstractVector,
)
    @. l.b = rand(GeneralizedInverseGaussian(1 / l.β^2, abs2(f - y), 0.5))
    set_ω!(l, inv.(l.b))
    return nothing
end

@inline ∇E_μ(
    l::LaplaceLikelihood{T},
    ::AOptimizer,
    y::AbstractVector,
) where {T} = (l.θ .* y,)
@inline ∇E_Σ(
    l::LaplaceLikelihood{T},
    ::AOptimizer,
    y::AbstractVector,
) where {T} = (0.5 * l.θ,)

## ELBO ##
function expec_log_likelihood(
    l::LaplaceLikelihood{T},
    i::AnalyticVI,
    y::AbstractVector,
    μ::AbstractVector,
    diag_cov::AbstractVector,
) where {T}
    tot = -0.5 * length(y) * log(twoπ)
    tot += 0.5 * sum(log, l.θ)
    tot +=
        -0.5 * (
            dot(l.θ, diag_cov) + dot(l.θ, abs2.(μ)) - 2.0 * dot(l.θ, μ .* y) +
            dot(l.θ, abs2.(y))
        )
    return tot
end

AugmentedKL(l::LaplaceLikelihood, ::AbstractVector) =
    GIGEntropy(l) - expecExponentialGIG(l)

GIGEntropy(l::LaplaceLikelihood{T}) where {T} = GIGEntropy(l.a, l.b, l.p)

function expecExponentialGIG(l::LaplaceLikelihood{T}) where {T}
    sum(
        -log(2 * l.β^2) .-
        0.5 * (l.a .* sqrt.(l.b) + l.b .* sqrt(l.a)) ./ (l.a .* l.b * l.β^2),
    )
end

## PDF and Log PDF Gradients ##

function grad_quad(
    likelihood::LaplaceLikelihood{T},
    y::Real,
    μ::Real,
    σ²::Real,
    inference::Inference,
) where {T<:Real}
    nodes = inference.nodes * sqrt(σ²) .+ μ
    Edlogpdf = dot(inference.weights, grad_logpdf.(likelihood, y, nodes))
    Ed²logpdf = (1 / sqrt(twoπ * σ²)) / (likelihood.β^2)
    return -Edlogpdf::T, Ed²logpdf::T
end


@inline grad_logpdf(l::LaplaceLikelihood{T}, y::Real, f::Real) where {T<:Real} =
    sign(y - f) ./ l.β

@inline hessian_logpdf(
    l::LaplaceLikelihood{T},
    y::Real,
    f::Real,
) where {T<:Real} = zero(T)
