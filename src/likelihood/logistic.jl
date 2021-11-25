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
    p(y|f,ω) = \exp\left(\frac{1}{2}(yf - (yf)^2 \omega)\right)
```
where ``ω \sim \mathcal{PG}(\omega | 1, 0)``, and ``\mathcal{PG}`` is the Polya-Gamma distribution.
See paper : [Efficient Gaussian Process Classification Using Polya-Gamma Data Augmentation](https://arxiv.org/abs/1802.06383).
"""
LogisticLikelihood() = BernoulliLikelihood(LogisticLink())

function implemented(
    ::BernoulliLikelihood{<:LogisticLink},
    ::Union{<:AnalyticVI,<:QuadratureVI,<:GibbsSampling},
)
    return true
end

function Distributions.loglikelihood(
    ::BernoulliLikelihood{<:LogisticLink}, y::Real, f::Real
)
    return -log(one(f) + exp(-y * f))
end

function compute_proba(l::BernoulliLikelihood, f::Real)
    return pdf(l(f), 1)
end

### Local Updates Section ###
function local_updates!(
    local_vars,
    ::BernoulliLikelihood{<:LogisticLink},
    ::AbstractVector,
    μ::AbstractVector,
    diagΣ::AbstractVector,
)
    map!(sqrt_expec_square, local_vars.c, μ, diagΣ) # √E[f^2]
    map!(local_vars.θ, local_vars.c) do c
        tanh(c / 2) / (2c)
    end # E[ω]
    return local_vars
end

function sample_local!(
    local_vars, ::BernoulliLikelihood{<:LogisticLink}, ::AbstractVector, f::AbstractVector
)
    map!(local_vars.θ, f) do f
        rand(PolyaGamma(1, abs(f)))
    end
    return local_vars
end

### Natural Gradient Section ###

function ∇E_μ(::BernoulliLikelihood{<:LogisticLink}, ::AOptimizer, y::AbstractVector, state)
    return (y / 2,)
end
function ∇E_Σ(::BernoulliLikelihood{<:LogisticLink}, ::AOptimizer, ::AbstractVector, state)
    return (state.θ / 2,)
end

### ELBO Section ###

function expec_loglikelihood(
    ::BernoulliLikelihood{<:LogisticLink},
    ::AnalyticVI,
    y::AbstractVector,
    μ::AbstractVector,
    diag_cov::AbstractVector,
    state,
)
    tot = -length(y) * logtwo / 2
    tot += (dot(μ, y) - dot(state.θ, diag_cov) - dot(state.θ, μ)) / 2
    return tot
end

AugmentedKL(l::BernoulliLikelihood{<:LogisticLink}, state, ::Any) = PolyaGammaKL(l, state)

function PolyaGammaKL(::BernoulliLikelihood{<:LogisticLink}, state)
    return sum(
        broadcast(PolyaGammaKL, ones(eltype(state.c), length(state.c)), state.c, state.θ)
    )
end

### Gradient Section ###

∇loglikehood(::BernoulliLikelihood{<:LogisticLink}, y::Real, f::Real) = y * logistic(-y * f)

function hessloglikehood(::BernoulliLikelihood{<:LogisticLink}, y::Real, f::Real)
    return -exp(y * f) / logistic(-y * f)^2
end
