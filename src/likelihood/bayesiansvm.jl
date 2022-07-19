@doc raw"""
    BayesianSVM() -> BernoulliLikelihood

The [Bayesian SVM](https://arxiv.org/abs/1707.05532) is a Bayesian interpretation of the classical SVM.
```math
p(y|f) \propto \exp(2 \max(1-yf, 0))
```

---

For the analytic version of the likelihood, it is augmented via:

```math
p(y|f, ω) = \frac{1}{\sqrt(2\pi\omega)} \exp\left(-\frac{(1+\omega-yf)^2}{2\omega})\right)
```

where ``ω \sim 1[0,\infty)`` has an improper prior (his posterior is however has a valid distribution, a Generalized Inverse Gaussian). For reference [see this paper](http://ecmlpkdd2017.ijs.si/papers/paperID502.pdf).
"""
BayesianSVM() = BernoulliLikelihood(SVMLink())

struct SVMLink <: AbstractLink end

implemented(::BernoulliLikelihood{<:SVMLink}, ::AnalyticVI) = true

function (::SVMLink)(f::Real)
    return svmlikelihood(f)
end

# Return likelihood equivalent to SVM hinge loss
function svmlikelihood(f::Real)
    pos = svmpseudolikelihood(f)
    return pos ./ (pos .+ svmpseudolikelihood(-f))
end

# Return the pseudo likelihood of the SVM hinge loss
function svmpseudolikelihood(f::Real)
    return exp(-2 * max(1 - f, zero(f)))
end

function local_updates!(
    local_vars,
    ::BernoulliLikelihood{<:SVMLink},
    y::AbstractVector,
    μ::AbstractVector{T},
    diagΣ::AbstractVector,
) where {T}
    map!(local_vars.c, μ, diagΣ, y) do μ, σ², y
        abs2(one(T) - y * μ) + σ²
    end
    map!(inv ∘ sqrt, local_vars.θ, local_vars.c)
    return local_vars
end

@inline function ∇E_μ(
    ::BernoulliLikelihood{<:SVMLink}, ::AOptimizer, y::AbstractVector, state
)
    return (y .* (state.θ .+ one(eltype(state.θ))),)
end

@inline function ∇E_Σ(
    ::BernoulliLikelihood{<:SVMLink}, ::AOptimizer, ::AbstractVector, state
)
    return (state.θ / 2,)
end

## ELBO

function expec_loglikelihood(
    ::BernoulliLikelihood{<:SVMLink},
    ::AnalyticVI,
    y,
    μ::AbstractVector,
    diag_cov::AbstractVector,
    state,
)
    tot = -length(y) * logtwo / 2
    tot += dot(μ, y)
    tot += -dot(state.θ, diag_cov) / 2 + dot(state.θ, abs2.(one(eltype(μ)) .- y .* μ))
    return tot
end

function AugmentedKL(l::BernoulliLikelihood{<:SVMLink}, state, ::Any)
    ChainRulesCore.@ignore_derivatives GIGEntropy(l, state)
end

function GIGEntropy(::BernoulliLikelihood{<:SVMLink}, state)
    return sum(log.(state.c)) / 2 + sum(log.(2.0 * besselk.(0.5, sqrt.(state.c)))) -
           sum(sqrt.(state.c)) / 2
end
