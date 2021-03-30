"""
    BayesianSVM()

The [Bayesian SVM](https://arxiv.org/abs/1707.05532) is a Bayesian interpretation of the classical SVM.
```math
p(y|f) \\propto \\exp(2 \\max(1-yf, 0))
````

---

For the analytic version of the likelihood, it is augmented via:

```math
p(y|f, ω) = \\frac{1}{\\sqrt(2\\pi\\omega) \\exp(-\\frac{(1+\\omega-yf)^2}{2\\omega}))
```

where ``ω ∼ 𝟙[0,∞)`` has an improper prior (his posterior is however has a valid distribution, a Generalized Inverse Gaussian). For reference [see this paper](http://ecmlpkdd2017.ijs.si/papers/paperID502.pdf)
"""
struct BayesianSVM{T<:Real,A<:AbstractVector{T}} <: ClassificationLikelihood{T}
    ω::A
    θ::A
    function BayesianSVM{T}() where {T<:Real}
        return new{T,Vector{T}}()
    end
    function BayesianSVM{T}(ω::A, θ::A) where {T<:Real,A<:AbstractVector{T}}
        return new{T,A}(ω, θ)
    end
end

function BayesianSVM()
    return BayesianSVM{Float64}()
end

implemented(::BayesianSVM, ::AnalyticVI) = true

function init_likelihood(
    ::BayesianSVM{T}, ::AbstractInference{T}, ::Int, nSamplesUsed::Int
) where {T}
    return BayesianSVM{T}(rand(T, nSamplesUsed), zeros(T, nSamplesUsed))
end
function (::BayesianSVM)(y::Real, f::Real)
    return svmlikelihood(y * f)
end

function Base.show(io::IO, ::BayesianSVM{T}) where {T}
    return print(io, "Bayesian SVM")
end

"""Return likelihood equivalent to SVM hinge loss"""
function svmlikelihood(f::Real)
    pos = svmpseudolikelihood(f)
    return pos ./ (pos .+ svmpseudolikelihood(-f))
end

"""Return the pseudo likelihood of the SVM hinge loss"""
function svmpseudolikelihood(f::Real)
    return exp(-2.0 * max.(1.0 - f, 0))
end

function compute_proba(
    ::BayesianSVM{T}, μ::AbstractVector{<:Real}, σ²::AbstractVector{<:Real}
) where {T<:Real}
    N = length(μ)
    pred = zeros(T, N)
    sig_pred = zeros(T, N)
    for i in 1:N
        x = pred_nodes .* sqrt(max(σ²[i], zero(T))) .+ μ[i]
        pred[i] = dot(pred_weights, svmlikelihood.(x))
        sig_pred[i] = max(dot(pred_weights, svmlikelihood.(x) .^ 2) - pred[i]^2, zero(T))
    end
    return pred, sig_pred
end

## Updates

function local_updates!(
    l::BayesianSVM{T}, y::AbstractVector, μ::AbstractVector, diagΣ::AbstractVector
) where {T}
    @. l.ω = abs2(one(T) - y * μ) + diagΣ
    @. l.θ = inv(sqrt(l.ω))
end

@inline function ∇E_μ(l::BayesianSVM{T}, ::AOptimizer, y::AbstractVector) where {T}
    return (y .* (l.θ .+ one(T)),)
end

@inline ∇E_Σ(l::BayesianSVM{T}, ::AOptimizer, ::AbstractVector) where {T} = (0.5 .* l.θ,)

## Lower bounds

function expec_loglikelihood(
    l::BayesianSVM{T},
    ::AnalyticVI,
    y::AbstractVector,
    μ::AbstractVector,
    diag_cov::AbstractVector,
) where {T}
    tot = -(0.5 * length(y) * logtwo)
    tot += dot(μ, y)
    tot += -0.5 * dot(l.θ, diag_cov) + dot(l.θ, abs2.(one(T) .- y .* μ))
    return tot
end

AugmentedKL(l::BayesianSVM, ::AbstractVector) = Zygote.@ignore(GIGEntropy(l))

function GIGEntropy(l::BayesianSVM)
    return 0.5 * sum(log.(l.ω)) + sum(log.(2.0 * besselk.(0.5, sqrt.(l.ω)))) -
           0.5 * sum(sqrt.(l.ω))
end
