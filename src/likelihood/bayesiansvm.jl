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
struct BayesianSVM{T} <: ClassificationLikelihood{T}
    ω::Vector{T}
    θ::Vector{T}
    function BayesianSVM{T}() where {T<:Real}
        new{T}()
    end
    function BayesianSVM{T}(
        ω::AbstractVector{<:Real},
        θ::AbstractVector{<:Real},
    ) where {T<:Real}
        new{T}(ω, θ)
    end
end

function BayesianSVM()
    BayesianSVM{Float64}()
end

implemented(::BayesianSVM, ::AnalyticVI) = true

function init_likelihood(
    likelihood::BayesianSVM{T},
    inference::Inference{T},
    nLatent::Integer,
    nSamplesUsed::Integer,
    nFeatures::Integer,
) where {T}
    BayesianSVM{T}(rand(T, nSamplesUsed), zeros(T, nSamplesUsed))
end
function pdf(l::BayesianSVM, y::Real, f::Real)
    svmlikelihood(y * f)
end

function Base.show(io::IO, model::BayesianSVM{T}) where {T}
    print(io, "Bayesian SVM")
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
    l::BayesianSVM{T},
    μ::AbstractVector,
    σ²::AbstractVector,
) where {T<:Real}
    N = length(μ)
    pred = zeros(T, N)
    sig_pred = zeros(T, N)
    for i = 1:N
        x = pred_nodes .* sqrt(max(σ²[i], zero(T))) .+ μ[i]
        pred[i] = dot(pred_weights, svmlikelihood.(x))
        sig_pred[i] =
            max(dot(pred_weights, svmlikelihood.(x) .^ 2) - pred[i]^2, zero(T))
    end
    return pred, sig_pred
end

## Updates

function local_updates!(
    l::BayesianSVM{T},
    y::AbstractVector,
    μ::AbstractVector,
    diag_cov::AbstractVector,
) where {T}
    l.ω .= abs2.(one(T) .- y .* μ) + diag_cov
    l.θ .= inv(sqrt.(l.ω))
end

@inline ∇E_μ(l::BayesianSVM{T}, ::AOptimizer, y::AbstractVector) where {T} =
    (y .* (l.θ .+ one(T)),)

@inline ∇E_Σ(l::BayesianSVM{T}, ::AOptimizer, y::AbstractVector) where {T} =
    (0.5 .* l.θ,)

## Lower bounds

function expec_log_likelihood(
    l::BayesianSVM{T},
    i::AnalyticVI,
    y::AbstractVector,
    μ::AbstractVector,
    diag_cov::AbstractVector,
) where {T}
    tot = -(0.5 * length(y) * logtwo)
    tot += dot(μ, y)
    tot += -0.5 * dot(l.θ, diag_cov) + dot(l.θ, abs2.(one(T) .- y .* μ))
    return tot
end

AugmentedKL(l::BayesianSVM, ::AbstractVector) = GIGEntropy(l)

function GIGEntropy(l::BayesianSVM)
    return 0.5 * sum(log.(l.ω)) + sum(log.(2.0 * besselk.(0.5, sqrt.(l.ω)))) -
           0.5 * sum(sqrt.(l.ω))
end
