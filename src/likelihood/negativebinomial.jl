"""
    NegBinomialLikelihood(r::Real)

## Arguments
- `r::Real` number of failures until the experiment is stopped

---
[Negative Binomial likelihood](https://en.wikipedia.org/wiki/Negative_binomial_distribution) with number of failures `r`
```math
    p(y|r, f) = binomial(y + r - 1, y) (1 - σ(f))ʳ σ(f)ʸ
    p(y|r, f) = Γ(y + r)/Γ(y + 1)Γ(r) (1 - σ(f))ʳ σ(f)ʸ
```
Where `σ` is the logistic function
"""
struct NegBinomialLikelihood{T<:Real,Tr<:Real,A<:AbstractVector{T}} <: EventLikelihood{T}
    r::Tr
    c::A
    θ::A
    function NegBinomialLikelihood{T}(r::Real) where {T<:Real}
        new{T,typeof(r),Vector{T}}(r)
    end
    function NegBinomialLikelihood{T}(
        r::Real,
        c::A,
        θ::A,
    ) where {T<:Real,A<:AbstractVector{T}}
        new{T,typeof(r),A}(r, c, θ)
    end
end

function NegBinomialLikelihood(r::Real)
    NegBinomialLikelihood{Float64}(r)
end

implemented(::NegBinomialLikelihood, ::Union{<:AnalyticVI,<:GibbsSampling}) =
    true

function init_likelihood(
    likelihood::NegBinomialLikelihood{T},
    ::AbstractInference{T},
    ::Int,
    nSamplesUsed::Int,
) where {T}
    NegBinomialLikelihood{T}(
        likelihood.r,
        rand(T, nSamplesUsed),
        zeros(T, nSamplesUsed),
    )
end

function (l::NegBinomialLikelihood)(y::Real, f::Real)
    pdf(NegativeBinomial(lr, get_p(l, f)), y)
end

function Distributions.loglikelihood(l::NegBinomialLikelihood, y::Real, f::Real)
    logpdf(NegativeBinomial(lr, get_p(l, f)), y)
end

function expec_count(l::NegBinomialLikelihood, f)
    broadcast((p, r) -> p * r ./ (1 .- p), get_p.(l, f), l.r)
end

function get_p(::NegBinomialLikelihood, f)
    logistic.(f)
end

function Base.show(io::IO, l::NegBinomialLikelihood{T}) where {T}
    print(io, "Negative Binomial Likelihood (r = $(l.r))")
end

function compute_proba(
    l::NegBinomialLikelihood{T},
    μ::AbstractVector{<:Real},
    σ²::AbstractVector{<:Real},
) where {T<:Real}
    N = length(μ)
    pred = zeros(T, N)
    sig_pred = zeros(T, N)
    for i = 1:N
        x = pred_nodes .* sqrt(max(σ²[i], zero(T))) .+ μ[i]
        pred[i] = dot(pred_weights, get_p.(l, x))
        sig_pred[i] = dot(pred_weights, get_p.(l, x) .^ 2) - pred[i]^2
    end
    return pred, sig_pred
end

## Local Updates ##

function local_updates!(
    l::NegBinomialLikelihood{T},
    y::AbstractVector,
    μ::AbstractVector,
    Σ::AbstractVector,
) where {T}
    @. l.c = sqrt(abs2(μ) + Σ)
    @. l.θ = (l.r + y) / l.c * tanh(0.5 * l.c)
end

function sample_local!(
    l::NegBinomialLikelihood,
    y::AbstractVector,
    f::AbstractVector,
)

    set_ω!(l, rand.(PolyaGamma.(y .+ Int(l.r), abs.(f))))
end

## Global Updates ##

@inline ∇E_μ(
    l::NegBinomialLikelihood{T},
    ::AOptimizer,
    y::AbstractVector,
) where {T} = (0.5 * (y .- l.r),)
@inline ∇E_Σ(
    l::NegBinomialLikelihood{T},
    ::AOptimizer,
    y::AbstractVector,
) where {T} = (0.5 .* l.θ,)

## ELBO Section ##

AugmentedKL(l::NegBinomialLikelihood, y::AbstractVector) =
    PolyaGammaKL(l, y)

function logabsbinomial(n, k)
    log(binomial(n, k))
end

function negbin_logconst(y, r::Real)
    loggamma.(y .+ r) - loggamma.(y .+ 1) .- loggamma(r)
end

function negbin_logconst(y, r::Int)
    logabsbinomial.(y .+ (r - 1), y)
end

function expec_loglikelihood(
    l::NegBinomialLikelihood{T},
    ::AnalyticVI,
    y::AbstractVector,
    μ::AbstractVector,
    diag_cov::AbstractVector,
) where {T}
    tot = Zygote.@ignore(sum(negbin_logconst(y, l.r))) - log(2.0) * sum(y .+ l.r)
    tot +=
        0.5 * dot(μ, (y .- l.r)) - 0.5 * dot(l.θ, μ) - 0.5 * dot(l.θ, diag_cov)
    return tot
end

function PolyaGammaKL(l::NegBinomialLikelihood, y::AbstractVector)
    PolyaGammaKL(y .+ l.r, l.c, l.θ)
end
