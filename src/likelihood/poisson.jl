"""
    Poisson Likelihood(λ=1.0)

[Poisson Likelihood](https://en.wikipedia.org/wiki/Poisson_distribution) where a Poisson distribution is defined at every point in space (careful, it's different from continous Poisson processes)
```math
    p(y|f) = Poisson(y|\\lambda \\sigma(f))
```
Where `σ` is the logistic function
Augmentation details will be released at some point (open an issue if you want to see them)
"""
mutable struct PoissonLikelihood{T<:Real,A<:AbstractVector{T}} <:
               EventLikelihood{T}
    λ::T
    c::A
    γ::A
    θ::A
    function PoissonLikelihood{T}(λ::T) where {T<:Real}
        new{T, Vector{T}}(λ)
    end
    function PoissonLikelihood{T}(
        λ::T,
        c::A,
        γ::A,
        θ::A,
    ) where {T<:Real,A<:AbstractVector{T}}
        new{T, A}(λ, c, γ, θ)
    end
end

function PoissonLikelihood(λ::T = 1.0) where {T<:Real}
    PoissonLikelihood{T}(λ)
end

implemented(::PoissonLikelihood, ::Union{<:AnalyticVI,<:GibbsSampling}) = true

function init_likelihood(
    likelihood::PoissonLikelihood{T},
    ::Inference{T},
    ::Integer,
    nSamplesUsed::Int,
) where {T}
    PoissonLikelihood{T}(
        likelihood.λ,
        zeros(T, nSamplesUsed),
        zeros(T, nSamplesUsed),
        zeros(T, nSamplesUsed),
    )
end

function (l::PoissonLikelihood)(y::Real, f::Real)
    pdf(Poisson(get_p(l, l.λ, f)), y)
end

function Distributions.loglikelihood(l::PoissonLikelihood, y::Real, f::Real)
    logpdf(Poisson(expec_count(l, f)), y)
end

function expec_count(l::PoissonLikelihood, f)
    get_p(l, l.λ, f)
end

function get_p(::PoissonLikelihood, λ::Real, f)
    λ * logistic.(f)
end

function Base.show(io::IO, l::PoissonLikelihood{T}) where {T}
    print(io, "Poisson Likelihood (λ = $(l.λ))")
end

function compute_proba(
    l::PoissonLikelihood{T},
    μ::AbstractVector{<:Real},
    σ²::AbstractVector{<:Real},
) where {T<:Real}
    N = length(μ)
    pred = zeros(T, N)
    sig_pred = zeros(T, N)
    for i = 1:N
        x = pred_nodes .* sqrt.(max(σ²[i], zero(T))) .+ μ[i]
        pred[i] = dot(pred_weights, get_p(l, l.λ, x))
        sig_pred[i] = dot(pred_weights, get_p(l, l.λ, x) .^ 2) - pred[i]^2
    end
    return pred, sig_pred
end

## Local Updates ##

function local_updates!(
    l::PoissonLikelihood{T},
    y::AbstractVector,
    μ::AbstractVector,
    diag_cov::AbstractVector,
) where {T}
    @. l.c = sqrt(abs2(μ) + diag_cov)
    @. l.γ = 0.5 * l.λ * safe_expcosh(-0.5 * μ, 0.5 * l.c)
    @. l.θ = (y + l.γ) / l.c * tanh(0.5 * l.c)
    l.λ = sum(y) / sum(expectation.(logistic, μ, diag_cov))
end

function sample_local!(
    l::PoissonLikelihood,
    y::AbstractVector,
    f::AbstractVector,
)
    @. l.γ = rand(Poisson(l.λ * logistic(f))) # Sample n
    pg = PolyaGammaDist()
    set_ω!(l, draw.([pg], y + l.γ, f)) # Sample ω
end

## Global Updates ##

@inline ∇E_μ(l::PoissonLikelihood, ::AOptimizer, y::AbstractVector) =
    (0.5 * (y - l.γ),)
@inline ∇E_Σ(l::PoissonLikelihood, ::AOptimizer, y::AbstractVector) =
    (0.5 * l.θ,)

## ELBO Section ##
function expec_loglikelihood(
    l::PoissonLikelihood{T},
    ::AnalyticVI,
    y,
    μ::AbstractVector,
    Σ::AbstractVector,
) where {T}
    tot = sum(y * log(l.λ)) - sum(logfactorial, y) - logtwo * sum((y + l.γ))
    tot += 0.5 * (dot(μ, (y - l.γ)) - dot(l.θ, abs2.(μ)) - dot(l.θ, Σ))
    return tot
end

AugmentedKL(l::PoissonLikelihood, y::AbstractVector) =
    PoissonKL(l) + PolyaGammaKL(l, y)

PoissonKL(l::PoissonLikelihood) = PoissonKL(l.γ, l.λ)

PolyaGammaKL(l::PoissonLikelihood, y::AbstractVector) =
    PolyaGammaKL(y + l.γ, l.c, l.θ)
