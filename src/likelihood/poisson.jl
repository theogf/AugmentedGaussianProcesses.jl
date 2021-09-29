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

function get_p(::PoissonLikelihood, λ::Real, f)
    return λ * logistic.(f)
end

function Base.show(io::IO, l::PoissonLikelihood)
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
        pred[i] = dot(pred_weights, get_p(l, l.λ, x))
        sig_pred[i] = dot(pred_weights, get_p(l, l.λ, x) .^ 2) - pred[i]^2
    end
    return pred, sig_pred
end

### Local Updates ###
function init_local_vars(::PoissonLikelihood, batchsize::Int, T::DataType=Float64)
    return (; c=rand(T, batchsize), θ=zeros(T, batchsize), γ=rand(T, batchsize))
end

function local_updates!(
    local_vars,
    l::PoissonLikelihood,
    y::AbstractVector,
    μ::AbstractVector,
    diag_cov::AbstractVector,
)
    @. local_vars.c = sqrt(abs2(μ) + diag_cov)
    @. local_vars.γ = 0.5 * l.λ * safe_expcosh(-0.5 * μ, 0.5 * local_vars.c)
    @. local_vars.θ = (y + local_vars.γ) / local_vars.c * tanh(0.5 * local_vars.c)
    l.λ = sum(y) / sum(expectation.(logistic, μ, diag_cov))
    return local_vars
end

function sample_local!(
    local_vars, l::PoissonLikelihood, y::AbstractVector, f::AbstractVector
)
    @. local_vars.γ = rand(Poisson(l.λ * logistic(f))) # Sample n
    @. local_vars.θ = rand(PolyaGamma(y + Int(local_vars.γ), abs(f))) # Sample ω
    return local_vars
end

### Global Updates ###

@inline function ∇E_μ(::PoissonLikelihood, ::AOptimizer, y::AbstractVector, state)
    return (0.5 * (y - state.γ),)
end
@inline ∇E_Σ(::PoissonLikelihood, ::AOptimizer, y::AbstractVector, state) = (0.5 * state.θ,)

## ELBO Section ##
function expec_loglikelihood(
    l::PoissonLikelihood, ::AnalyticVI, y, μ::AbstractVector, Σ::AbstractVector, state
)
    tot = 0.5 * (dot(μ, (y - state.γ)) - dot(state.θ, abs2.(μ)) - dot(state.θ, Σ))
    tot += Zygote.@ignore(
        sum(y * log(l.λ)) - sum(logfactorial, y) - logtwo * sum((y + state.γ))
    )
    return tot
end

function AugmentedKL(l::PoissonLikelihood, state, y)
    return PoissonKL(l, state) + PolyaGammaKL(l, state, y)
end

PoissonKL(l::PoissonLikelihood, state) = PoissonKL(state.γ, l.λ)

function PolyaGammaKL(::PoissonLikelihood, state, y)
    return PolyaGammaKL(y + state.γ, state.c, state.θ)
end
