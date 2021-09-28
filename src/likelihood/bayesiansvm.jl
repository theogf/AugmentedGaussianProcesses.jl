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

struct SVMLink <: Link end

implemented(::BernoulliLikelihood{<:SVMLink}, ::AnalyticVI) = true

function (::SVMLink)(f::Real)
    return svmlikelihood(f)
end

function Base.show(io::IO, ::BayesianSVM{T}) where {T}
    return print(io, "Bayesian SVM")
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
    ::BernoulliLikelihood{<:SVMLink,T},
    y::AbstractVector,
    μ::AbstractVector,
    diagΣ::AbstractVector,
) where {T}
    @. local_vars.c = abs2(one(T) - y * μ) + diagΣ
    @. local_vars.θ = inv(sqrt(local_vars.c))
    return local_vars
end

@inline function ∇E_μ(::BernoulliLikelihood{<:SVMLink,T}, ::AOptimizer, y::AbstractVector, state) where {T}
    return (y .* (state.θ .+ one(T)),)
end

@inline function ∇E_Σ(::BernoulliLikelihood{<:SVMLink,T}, ::AOptimizer, ::AbstractVector, state) where {T}
    return (0.5 .* state.θ,)
end

## ELBO

function expec_loglikelihood(
    ::BernoulliLikelihood{<:SVMLink,T}, ::AnalyticVI, y, μ::AbstractVector, diag_cov::AbstractVector, state
) where {T}
    tot = -(0.5 * length(y) * logtwo)
    tot += dot(μ, y)
    tot += -0.5 * dot(state.θ, diag_cov) + dot(state.θ, abs2.(one(T) .- y .* μ))
    return tot
end

AugmentedKL(l::BernoulliLikelihood{<:SVMLink}, state, ::Any) = Zygote.@ignore(GIGEntropy(l, state))

function GIGEntropy(::BernoulliLikelihood{<:SVMLink}, state)
    return 0.5 * sum(log.(state.c)) + sum(log.(2.0 * besselk.(0.5, sqrt.(state.c)))) -
           0.5 * sum(sqrt.(state.c))
end
