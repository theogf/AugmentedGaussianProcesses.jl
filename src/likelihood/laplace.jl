@doc raw"""
    LaplaceLikelihood(β::T=1.0)  #  Laplace likelihood with scale β

Laplace likelihood for regression:
```math
\frac{1}{2\beta} \exp(-\frac{|y-f|}{β})
```
see [wiki page](https://en.wikipedia.org/wiki/Laplace_distribution)
---
For the analytical solution, it is augmented via:
```math
p(y|f,ω) = N(y|f,ω⁻¹)
```
where ``ω \sim \text{Exp}(ω | 1/(2 β^2))``, and ``\text{Exp}`` is the [Exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution)
We use the variational distribution ``q(ω) = GIG(ω|a,b,p)``
"""
struct LaplaceLikelihood{T<:Real} <: RegressionLikelihood
    β::T
    a::T
    p::T
    function LaplaceLikelihood{T}(β::T) where {T<:Real}
        return new{T}(β, β^-2, 0.5)
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

function Base.show(io::IO, l::LaplaceLikelihood)
    return print(io, "Laplace likelihood (β=$(l.β))")
end

function compute_proba(
    l::LaplaceLikelihood, μ::AbstractVector{<:Real}, σ²::AbstractVector{<:Real}
)
    return μ, max.(σ², zero(eltype(σ²))) .+ 2 * l.β^2
end

## Local Updates ##
# b : Variational parameter b of GIG
# θ : Expected value of ω
function init_local_vars(::LaplaceLikelihood, batchsize::Int, T::DataType=Float64)
    return (; b=rand(T, batchsize), θ=zeros(T, batchsize))
end

function local_updates!(
    local_vars,
    l::LaplaceLikelihood,
    y::AbstractVector,
    μ::AbstractVector,
    diagΣ::AbstractVector,
)
    map!(sqrt_expec_square, local_vars.b, μ, diagΣ, y) # √E[(f-y)^2]
    map!(local_vars.θ, local_vars.b) do b
        sqrt(l.a) / b
    end
    return local_vars
end

function sample_local!(
    local_vars, l::LaplaceLikelihood, y::AbstractVector, f::AbstractVector
)
    map!(local_vars.b, f, y) do f, y
        rand(GeneralizedInverseGaussian(inv(l.β^2), abs2(f - y), 0.5))
    end
    map!(inv, local_vars.θ, local_vars.b)
    return local_vars
end

@inline function ∇E_μ(::LaplaceLikelihood, ::AOptimizer, y::AbstractVector, state)
    return (state.θ .* y,)
end
@inline function ∇E_Σ(::LaplaceLikelihood, ::AOptimizer, ::AbstractVector, state)
    return (state.θ / 2,)
end

## ELBO ##
function expec_loglikelihood(
    ::LaplaceLikelihood,
    ::AnalyticVI,
    y::AbstractVector,
    μ::AbstractVector,
    diag_cov::AbstractVector,
    state,
)
    tot = -length(y) * log(twoπ) / 2
    tot += ChainRulesCore.@ignore_derivatives sum(log, state.θ) / 2
    tot +=
        -(
            dot(state.θ, diag_cov) + dot(state.θ, abs2.(μ)) - 2.0 * dot(state.θ, μ .* y) +
            dot(state.θ, abs2.(y))
        ) / 2
    return tot
end

function AugmentedKL(l::LaplaceLikelihood, state, ::Any)
    return GIGEntropy(l, state) - expecExponentialGIG(l, state)
end

GIGEntropy(l::LaplaceLikelihood, state) = GIGEntropy(l.a, state.b .^ 2, l.p)

function expecExponentialGIG(l::LaplaceLikelihood, state)
    return sum(
        -log(2 * l.β^2) .-
        (l.a .* state.b + state.b .^ 2 .* sqrt(l.a)) ./ (l.a .* state.b .^ 2 * l.β^2) / 2,
    )
end

## PDF and Log PDF Gradients ##

function grad_quad(
    likelihood::LaplaceLikelihood, y::Real, μ::Real, σ²::Real, inference::AbstractInference
)
    nodes = inference.nodes * sqrt(σ²) .+ μ
    Edloglike = dot(inference.weights, ∇loglikehood.(likelihood, y, nodes))
    Ed²loglike = (1 / sqrt(twoπ * σ²)) / (likelihood.β^2)
    return -Edloglike, Ed²loglike
end

@inline function ∇loglikehood(l::LaplaceLikelihood, y::Real, f::Real)
    return sign(y - f) ./ l.β
end

@inline hessloglikelihood(::LaplaceLikelihood, ::Real, f::T) where {T<:Real} = zero(T)
