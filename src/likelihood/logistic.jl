@doc raw"""
    LogisticLikelihood() -> BernoulliLikelihood

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
LogisticLikelihood() = BernoulliLikelihood(LogisticLink())

function implemented(
    ::BernoulliLikelihood{<:LogisticLink}, ::Union{<:AnalyticVI,<:QuadratureVI,<:GibbsSampling}
)
    return true
end

function Distributions.loglikelihood(::BernoulliLikelihood{<:LogisticLink,T}, y::Real, f::Real) where {T}
    return -log(one(T) + exp(-y * f))
end

function compute_proba(l::LogisticLikelihood{T}, f::Real) where {T<:Real}
    return pdf(l(f), 1)
end

### Local Updates Section ###
function init_local_vars(state, ::BernoulliLikelihood{<:LogisticLink,T}, batchsize::Int) where {T}
    return merge(state, (; local_vars=(; c=rand(T, batchsize), θ=zeros(T, batchsize))))
end

function local_updates!(
    local_vars,
    ::BernoulliLikelihood{<:LogisticLink},
    ::AbstractVector,
    μ::AbstractVector,
    diagΣ::AbstractVector,
)
    @. local_vars.c = sqrt(diagΣ + abs2(μ))
    @. local_vars.θ = 0.5 * tanh(0.5 * local_vars.c) / local_vars.c
    return local_vars
end

function sample_local!(
    local_vars, ::BernoulliLikelihood{<:LogisticLink}, ::AbstractVector, f::AbstractVector
)
    local_vars.θ .= rand.(PolyaGamma.(1, abs.(f)))
    return local_vars
end

### Natural Gradient Section ###

∇E_μ(::BernoulliLikelihood{<:LogisticLink}, ::AOptimizer, y::AbstractVector, state) = (0.5 * y,)
∇E_Σ(::BernoulliLikelihood{<:LogisticLink}, ::AOptimizer, ::AbstractVector, state) = (0.5 * state.θ,)

### ELBO Section ###

function expec_loglikelihood(
    ::BernoulliLikelihood{<:LogisticLink},
    ::AnalyticVI,
    y::AbstractVector,
    μ::AbstractVector,
    diag_cov::AbstractVector,
    state,
)
    tot = -(0.5 * length(y) * logtwo)
    tot += 0.5 .* (dot(μ, y) - dot(state.θ, diag_cov) - dot(state.θ, μ))
    return tot
end

AugmentedKL(l::BernoulliLikelihood{<:LogisticLink}, state, ::Any) = PolyaGammaKL(l, state)

function PolyaGammaKL(::BernoulliLikelihood{<:LogisticLink,T}, state) where {T}
    return sum(broadcast(PolyaGammaKL, ones(T, length(state.c)), state.c, state.θ))
end

### Gradient Section ###

∇loglikehood(::BernoulliLikelihood{<:LogisticLink}, y::Real, f::Real) = y * logistic(-y * f)

function hessloglikehood(::BernoulliLikelihood{<:LogisticLink}, y::Real, f::Real)
    return -exp(y * f) / logistic(-y * f)^2
end
