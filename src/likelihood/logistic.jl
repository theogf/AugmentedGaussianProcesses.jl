@doc raw"""
    LogisticLikelihood()

Bernoulli likelihood with a logistic link for the Bernoulli likelihood
```math
    p(y|f) = \sigma(yf) = \frac{1}{1 + \exp(-yf)},
```
(for more info see : [wiki page](https://en.wikipedia.org/wiki/Logistic_function))

---

For the analytic version the likelihood, it is augmented via:
```math
    p(y|f,ω) = \exp\left(0.5(yf - (yf)^2 \omega)\right)
```
where ``ω \sim \mathcal{PG}(\omega | 1, 0)``, and ``\mathcal{PG}`` is the Polya-Gamma distribution.
See paper : [Efficient Gaussian Process Classification Using Polya-Gamma Data Augmentation](https://arxiv.org/abs/1802.06383).
"""
struct LogisticLikelihood{T<:Real} <: ClassificationLikelihood{T} end

function LogisticLikelihood()
    return LogisticLikelihood{Float64}()
end

function implemented(
    ::LogisticLikelihood, ::Union{<:AnalyticVI,<:QuadratureVI,<:GibbsSampling}
)
    return true
end

function (::LogisticLikelihood)(y::Real, f::Real)
    return logistic(y * f)
end

function Distributions.loglikelihood(::LogisticLikelihood{T}, y::Real, f::Real) where {T}
    return -log(one(T) + exp(-y * f))
end

function Base.show(io::IO, ::LogisticLikelihood)
    return print(io, "Bernoulli Likelihood with Logistic Link")
end

function compute_proba(l::LogisticLikelihood{T}, f::Real) where {T<:Real}
    return l(1, f)
end

function compute_proba(
    ::LogisticLikelihood{T}, μ::AbstractVector{<:Real}, σ²::AbstractVector{<:Real}
) where {T<:Real}
    N = length(μ)
    pred = zeros(T, N)
    σ²_pred = zeros(T, N)
    for i in 1:N
        x = pred_nodes .* sqrt(max(σ²[i], zero(T))) .+ μ[i]
        pred[i] = dot(pred_weights, logistic.(x))
        σ²_pred[i] = max(dot(pred_weights, logistic.(x) .^ 2) - pred[i]^2, zero(T))
    end
    return pred, σ²_pred
end

### Local Updates Section ###
function init_local_vars(state, ::LogisticLikelihood{T}, batchsize::Int) where {T}
    return merge(state, (; local_vars=(; c=rand(T, batchsize), θ=zeros(T, batchsize))))
end

function local_updates!(
    local_vars,
    ::LogisticLikelihood{T},
    ::AbstractVector,
    μ::AbstractVector,
    diagΣ::AbstractVector,
) where {T}
    @. local_vars.c = sqrt(diagΣ + abs2(μ))
    @. local_vars.θ = 0.5 * tanh(0.5 * local_vars.c) / local_vars.c
    return local_vars
end

function sample_local!(local_vars, ::LogisticLikelihood, ::AbstractVector, f::AbstractVector)
    local_vars.θ .= rand.(PolyaGamma.(1, abs.(f)))
    return local_vars
end

### Natural Gradient Section ###

∇E_μ(::LogisticLikelihood, ::AOptimizer, y::AbstractVector, state) = (0.5 * y,)
∇E_Σ(::LogisticLikelihood, ::AOptimizer, ::AbstractVector, state) = (0.5 * state.θ,)

### ELBO Section ###

function expec_loglikelihood(
    ::LogisticLikelihood{T},
    ::AnalyticVI,
    y::AbstractVector,
    μ::AbstractVector,
    diag_cov::AbstractVector,
    state,
) where {T}
    tot = -(0.5 * length(y) * logtwo)
    tot += 0.5 .* (dot(μ, y) - dot(state.θ, diag_cov) - dot(state.θ, μ))
    return tot
end

AugmentedKL(l::LogisticLikelihood, state, ::Any) = PolyaGammaKL(l, state)

function PolyaGammaKL(::LogisticLikelihood{T}, state) where {T}
    return sum(broadcast(PolyaGammaKL, ones(T, length(state.c)), state.c, state.θ))
end

### Gradient Section ###

∇loglikehood(::LogisticLikelihood{T}, y::Real, f::Real) where {T} = y * logistic(-y * f)

function hessloglikehood(::LogisticLikelihood{T}, y::Real, f::Real) where {T<:Real}
    return -exp(y * f) / logistic(-y * f)^2
end
