@doc raw"""
    PoissonLikelihood(λ::Real)->PoissonLikelihood

## Arguments
- `λ::Real` : Maximal Poisson rate

---

[Poisson Likelihood](https://en.wikipedia.org/wiki/Poisson_distribution) where a Poisson distribution is defined at every point in space (careful, it's different from continous Poisson processes).
```math
    p(y|f) = \text{Poisson}(y|\lambda \sigma(f))
```
Where ``\sigma`` is the logistic function.
Augmentation details will be released at some point (open an issue if you want to see them)
"""
function PoissonLikelihood(λ::Real)
    return PoissonLikelihood(ScaledLogistic(λ))
end

struct ScaledLogistic{T} <: AbstractLink
    λ::Vector{T}
end

ScaledLogistic(λ::Real) = ScaledLogistic([λ])

(l::ScaledLogistic)(f::Real) = l.λ[1] * logistic(f)

function implemented(
    ::PoissonLikelihood{<:ScaledLogistic}, ::Union{<:AnalyticVI,<:GibbsSampling}
)
    return true
end

function (l::PoissonLikelihood)(y::Real, f::Real)
    return pdf(l(f), y)
end

function Distributions.loglikelihood(l::PoissonLikelihood, y::Real, f::Real)
    return logpdf(l(f), y)
end

function Base.show(io::IO, l::PoissonLikelihood{<:ScaledLogistic})
    return print(io, "Poisson Likelihood (λ = $(l.invlink.λ[1]))")
end

function compute_proba(
    l::PoissonLikelihood, μ::AbstractVector{T}, σ²::AbstractVector{<:Real}
) where {T}
    N = length(μ)
    pred = zeros(T, N)
    sig_pred = zeros(T, N)
    for i in 1:N
        x = pred_nodes .* sqrt.(max(σ²[i], zero(σ²[i]))) .+ μ[i]
        pred[i] = dot(pred_weights, l.invlink.(x))
        sig_pred[i] = dot(pred_weights, l.invlink.(x) .^ 2) - pred[i]^2
    end
    return pred, sig_pred
end

### Local Updates ###
function init_local_vars(::PoissonLikelihood, batchsize::Int, T::DataType=Float64)
    return (; c=rand(T, batchsize), θ=zeros(T, batchsize), γ=rand(T, batchsize))
end

function local_updates!(
    local_vars,
    l::PoissonLikelihood{<:ScaledLogistic},
    y::AbstractVector,
    μ::AbstractVector,
    diagΣ::AbstractVector,
)
    λ = only(l.invlink.λ)
    map!(sqrt_expec_square, local_vars.c, μ, diagΣ)
    map!(local_vars.γ, μ, local_vars.c) do μ, c
        λ * safe_expcosh(-μ / 2, c / 2) / 2
    end
    map!(local_vars.θ, y, local_vars.γ, local_vars.c) do y, γ, c
        (y + γ) / c * tanh(c / 2)
    end
    l.invlink.λ .= sum(y) / sum(expectation.(logistic, μ, diagΣ))
    return local_vars
end

function sample_local!(
    local_vars, l::PoissonLikelihood{<:ScaledLogistic}, y::AbstractVector, f::AbstractVector
)
    map!(rand ∘ l, local_vars.γ, f) # sample n
    map!(local_vars.θ, y, local_vars.γ, f) do y, γ, f
        rand(PolyaGamma(y + Int(γ), abs(f))) # Sample ω
    end
    return local_vars
end

### Global Updates ###

@inline function ∇E_μ(
    ::PoissonLikelihood{<:ScaledLogistic}, ::AOptimizer, y::AbstractVector, state
)
    return ((y - state.γ) / 2,)
end
@inline function ∇E_Σ(
    ::PoissonLikelihood{<:ScaledLogistic}, ::AOptimizer, y::AbstractVector, state
)
    return (state.θ / 2,)
end

## ELBO Section ##
function expec_loglikelihood(
    l::PoissonLikelihood{<:ScaledLogistic},
    ::AnalyticVI,
    y,
    μ::AbstractVector,
    Σ::AbstractVector,
    state,
)
    tot = (dot(μ, (y - state.γ)) - dot(state.θ, abs2.(μ)) - dot(state.θ, Σ)) / 2
    tot += ChainRulesCore.ignore_derivatives() do
        sum(y * log(l.invlink.λ[1])) - sum(logfactorial, y) - logtwo * sum((y + state.γ))
    end
    return tot
end

function AugmentedKL(l::PoissonLikelihood{<:ScaledLogistic}, state, y)
    return PoissonKL(l, state) + PolyaGammaKL(l, state, y)
end

function PoissonKL(l::PoissonLikelihood{<:ScaledLogistic}, state)
    return PoissonKL(state.γ, only(l.invlink.λ))
end

function PolyaGammaKL(::PoissonLikelihood{<:ScaledLogistic}, state, y)
    return PolyaGammaKL(y + state.γ, state.c, state.θ)
end
