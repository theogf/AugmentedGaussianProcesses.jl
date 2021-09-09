"""
    LaplaceLikelihood(β::T=1.0)  #  Laplace likelihood with scale β

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
struct LaplaceLikelihood{T<:Real} <: RegressionLikelihood{T}
    β::T
    a::T
    p::T
    function LaplaceLikelihood{T}(β::T) where {T<:Real}
        return new{T,Vector{T}}(β, β^-2, 0.5)
    end
end

function LaplaceLikelihood(β::T=1.0) where {T<:Real}
    return LaplaceLikelihood{T}(β)
end

function implemented(
    ::LaplaceLikelihood, ::Union{<:AnalyticVI,<:QuadratureVI,<:GibbsSampling}
)
    return true
end

function (l::LaplaceLikelihood)(y::Real, f::Real)
    return pdf(Laplace(f, l.β), y)
end

function Distributions.loglikelihood(l::LaplaceLikelihood, y::Real, f::Real)
    return logpdf(Laplace(f, l.β), y)
end

function Base.show(io::IO, l::LaplaceLikelihood{T}) where {T}
    return print(io, "Laplace likelihood (β=$(l.β))")
end

function compute_proba(
    l::LaplaceLikelihood{T}, μ::AbstractVector{<:Real}, σ²::AbstractVector{<:Real}
) where {T<:Real}
    return μ, max.(σ², 0.0) .+ 2 * l.β^2
end

## Local Updates ##
# b : Variational parameter b of GIG
# θ : Expected value of ω
function init_local_vars_state(state, ::LaplaceLikelihood{T}, batchsize::Int)
    return merge(state, (; local_vars=(; b=rand(T, batchsize), θ=zeros(T, batchsize))))
end

function local_updates!(
    state,
    l::LaplaceLikelihood{T},
    y::AbstractVector,
    μ::AbstractVector,
    diagΣ::AbstractVector,
) where {T}
    @. state.b = diagΣ + abs2(μ - y)
    @. state.θ = sqrt(l.a) / sqrt.(l.b)
    return state
end

function sample_local!(l::LaplaceLikelihood, y::AbstractVector, f::AbstractVector)
    @. l.b = rand(GeneralizedInverseGaussian(1 / l.β^2, abs2(f - y), 0.5))
    set_ω!(l, inv.(l.b))
    return nothing
end

@inline function ∇E_μ(
    ::LaplaceLikelihood{T}, ::AOptimizer, y::AbstractVector, state
) where {T}
    return (state.θ .* y,)
end
@inline function ∇E_Σ(
    ::LaplaceLikelihood{T}, ::AOptimizer, ::AbstractVector, state
) where {T}
    return (0.5 * state.θ,)
end

## ELBO ##
function expec_loglikelihood(
    l::LaplaceLikelihood{T},
    ::AnalyticVI,
    y::AbstractVector,
    μ::AbstractVector,
    diag_cov::AbstractVector,
    state,
) where {T}
    tot = -0.5 * length(y) * log(twoπ)
    tot += 0.5 * Zygote.@ignore(sum(log, state.θ))
    tot +=
        -0.5 * (
            dot(state.θ, diag_cov) + dot(state.θ, abs2.(μ)) - 2.0 * dot(state.θ, μ .* y) +
            dot(state.θ, abs2.(y))
        )
    return tot
end

function AugmentedKL(l::LaplaceLikelihood, ::AbstractVector, state)
    return GIGEntropy(l, state) - expecExponentialGIG(l, state)
end

GIGEntropy(l::LaplaceLikelihood{T}, state) where {T} = GIGEntropy(l.a, state.b, l.p)

function expecExponentialGIG(l::LaplaceLikelihood{T}, state) where {T}
    return sum(
        -log(2 * l.β^2) .-
        0.5 * (l.a .* sqrt.(state.b) + state.b .* sqrt(l.a)) ./ (l.a .* state.b * l.β^2),
    )
end

## PDF and Log PDF Gradients ##

function grad_quad(
    likelihood::LaplaceLikelihood{T},
    y::Real,
    μ::Real,
    σ²::Real,
    inference::AbstractInference,
) where {T<:Real}
    nodes = inference.nodes * sqrt(σ²) .+ μ
    Edloglike = dot(inference.weights, ∇loglikehood.(likelihood, y, nodes))
    Ed²loglike = (1 / sqrt(twoπ * σ²)) / (likelihood.β^2)
    return -Edloglike::T, Ed²loglike::T
end

@inline function ∇loglikehood(l::LaplaceLikelihood{T}, y::Real, f::Real) where {T<:Real}
    return sign(y - f) ./ l.β
end

@inline hessloglikelihood(::LaplaceLikelihood{T}, ::Real, ::Real) where {T<:Real} = zero(T)
