"""
```julia
LogisticLikelihood()
```

Bernoulli likelihood with a logistic link for the Bernoulli likelihood
```math
    p(y|f) = \\sigma(yf) = \\frac{1}{1+\\exp(-yf)},
```
(for more info see : [wiki page](https://en.wikipedia.org/wiki/Logistic_function))

---

For the analytic version the likelihood, it is augmented via:
```math
    p(y|f,ω) = exp(0.5(yf - (yf)^2 ω))
```
where ``ω ~ PG(ω | 1, 0)``, and `PG` is the Polya-Gamma distribution
See paper : [Efficient Gaussian Process Classification Using Polya-Gamma Data Augmentation](https://arxiv.org/abs/1802.06383)
"""
struct LogisticLikelihood{T<:Real, A<:AbstractVector{T}} <: ClassificationLikelihood{T}
    c::A
    θ::A
    function LogisticLikelihood{T}() where {T<:Real}
        new{T, Vector{T}}()
    end
    function LogisticLikelihood{T}(
        c::A,
        θ::A,
    ) where {T<:Real, A<:AbstractVector{T}}
        new{T, A}(c, θ)
    end
end

function LogisticLikelihood()
    LogisticLikelihood{Float64}()
end

implemented(
    ::LogisticLikelihood,
    ::Union{<:AnalyticVI,<:QuadratureVI,<:GibbsSampling},
) = true

function init_likelihood(
    likelihood::LogisticLikelihood{T},
    inference::Inference{T},
    nLatent::Int,
    nSamplesUsed::Int,
) where {T}
    if inference isa AnalyticVI || inference isa GibbsSampling
        LogisticLikelihood{T}(
            abs.(rand(T, nSamplesUsed)),
            zeros(T, nSamplesUsed),
        )
    else
        LogisticLikelihood{T}()
    end
end

function pdf(l::LogisticLikelihood, y::Real, f::Real)
    logistic(y * f)
end

function logpdf(l::LogisticLikelihood, y::T, f::T) where {T<:Real}
    -log(one(T) + exp(-y * f))
end

function Base.show(io::IO, ::LogisticLikelihood)
    print(io, "Bernoulli Likelihood with Logistic Link")
end

function compute_proba(
    l::LogisticLikelihood{T},
    f::Real
    ) where {T<:Real}
    pdf(l, 1, f)
end

function compute_proba(
    l::LogisticLikelihood{T},
    μ::AbstractVector{<:Real},
    σ²::AbstractVector{<:Real},
) where {T<:Real}
    N = length(μ)
    pred = zeros(T, N)
    σ²_pred = zeros(T, N)
    for i = 1:N
        x = pred_nodes .* sqrt(max(σ²[i], zero(T))) .+ μ[i]
        pred[i] = dot(pred_weights, logistic.(x))
        σ²_pred[i] =
            max(dot(pred_weights, logistic.(x) .^ 2) - pred[i]^2, zero(T))
    end
    return pred, σ²_pred
end

### Local Updates Section ###

function local_updates!(
    l::LogisticLikelihood{T},
    ::AbstractVector,
    μ::AbstractVector,
    diagΣ::AbstractVector,
) where {T}
    @. l.c = sqrt(diagΣ + abs2(μ))
    @. l.θ = 0.5 * tanh(0.5 * l.c) / l.c
end

function sample_local!(
    l::LogisticLikelihood,
    ::AbstractVector,
    f::AbstractVector,
)
    pg = PolyaGammaDist()
    set_ω!(l, draw.([pg], [1.0], f))
end

### Natural Gradient Section ###

∇E_μ(l::LogisticLikelihood, ::AOptimizer, y::AbstractVector) = (0.5 * y,)
∇E_Σ(l::LogisticLikelihood, ::AOptimizer, y::AbstractVector) = (0.5 * l.θ,)

### ELBO Section ###

function expec_log_likelihood(
    l::LogisticLikelihood{T},
    i::AnalyticVI,
    y::AbstractVector,
    μ::AbstractVector,
    diag_cov::AbstractVector,
) where {T}
    tot = -(0.5 * length(y) * logtwo)
    tot += 0.5 .* (dot(μ, y) - dot(l.θ, diag_cov) - dot(l.θ, μ))
    return tot
end

AugmentedKL(l::LogisticLikelihood, ::AbstractVector) = PolyaGammaKL(l)

function PolyaGammaKL(l::LogisticLikelihood{T}) where {T}
    sum(broadcast(PolyaGammaKL, ones(T, length(l.c)), l.c, l.θ))
end

### Gradient Section ###

@inline grad_logpdf(::LogisticLikelihood{T}, y::Real, f::Real) where {T} =
    y * logistic(-y * f)

@inline hessian_logpdf(
    ::LogisticLikelihood{T},
    y::Real,
    f::Real,
) where {T<:Real} = -exp(y * f) / logistic(-y * f)^2
