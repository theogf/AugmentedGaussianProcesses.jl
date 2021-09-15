@doc raw"""
    NegBinomialLikelihood(r::Real)

## Arguments
- `r::Real` number of failures until the experiment is stopped

---

[Negative Binomial likelihood](https://en.wikipedia.org/wiki/Negative_binomial_distribution) with number of failures `r`
```math
    p(y|r, f) = {y + r - 1 \choose y} (1 - \sigma(f))^r \sigma(f)^y,
```
if ``r\in \mathbb{N}`` or
```math
    p(y|r, f) = \frac{\Gamma(y + r)}{\Gamma(y + 1)\Gamma(r)} (1 - \sigma(f))^r \sigma(f)^y,
```
if ``r\in\mathbb{R}``.
Where ``\sigma`` is the logistic function
"""
struct NegBinomialLikelihood{T<:Real} <: EventLikelihood{T}
    r::T
end

implemented(::NegBinomialLikelihood, ::Union{<:AnalyticVI,<:GibbsSampling}) = true

function (l::NegBinomialLikelihood)(y::Real, f::Real)
    return pdf(NegativeBinomial(lr, get_p(l, f)), y)
end

function Distributions.loglikelihood(l::NegBinomialLikelihood, y::Real, f::Real)
    return logpdf(NegativeBinomial(lr, get_p(l, f)), y)
end

function expec_count(l::NegBinomialLikelihood, f)
    return broadcast((p, r) -> p * r ./ (1 .- p), get_p.(l, f), l.r)
end

function get_p(::NegBinomialLikelihood, f)
    return logistic.(f)
end

function Base.show(io::IO, l::NegBinomialLikelihood{T}) where {T}
    return print(io, "Negative Binomial Likelihood (r = $(l.r))")
end

function compute_proba(
    l::NegBinomialLikelihood{T}, μ::AbstractVector{<:Real}, σ²::AbstractVector{<:Real}
) where {T<:Real}
    N = length(μ)
    pred = zeros(T, N)
    sig_pred = zeros(T, N)
    for i in 1:N
        x = pred_nodes .* sqrt(max(σ²[i], zero(T))) .+ μ[i]
        pred[i] = dot(pred_weights, get_p.(l, x))
        sig_pred[i] = dot(pred_weights, get_p.(l, x) .^ 2) - pred[i]^2
    end
    return pred, sig_pred
end

## Local Updates ##
function init_local_vars(state, ::NegBinomialLikelihood{T}, batchsize::Int) where {T}
    return merge(state, (; local_vars=(; c=rand(T, batchsize), θ=zeros(T, batchsize))))
end


function local_updates!(
    local_vars, l::NegBinomialLikelihood{T}, y::AbstractVector, μ::AbstractVector, Σ::AbstractVector
) where {T}
    @. local_vars.c = sqrt(abs2(μ) + Σ)
    @. local_vars.θ = (l.r + y) / local_vars.c * tanh(0.5 * local_vars.c)
    return local_vars
end

function sample_local!(l::NegBinomialLikelihood, y::AbstractVector, f::AbstractVector)
    return set_ω!(l, rand.(PolyaGamma.(y .+ Int(l.r), abs.(f))))
end

## Global Updates ##

@inline function ∇E_μ(
    l::NegBinomialLikelihood{T}, ::AOptimizer, y::AbstractVector, state
) where {T}
    return (0.5 * (y .- l.r),)
end
@inline function ∇E_Σ(
    ::NegBinomialLikelihood{T}, ::AOptimizer, y::AbstractVector, state
) where {T}
    return (0.5 .* state.θ,)
end

## ELBO Section ##

AugmentedKL(l::NegBinomialLikelihood, y::AbstractVector, state) = PolyaGammaKL(l, y, state)

function logabsbinomial(n, k)
    return log(binomial(n, k))
end

function negbin_logconst(y, r::Real)
    return loggamma.(y .+ r) - loggamma.(y .+ 1) .- loggamma(r)
end

function negbin_logconst(y, r::Int)
    return logabsbinomial.(y .+ (r - 1), y)
end

function expec_loglikelihood(
    l::NegBinomialLikelihood{T},
    ::AnalyticVI,
    y::AbstractVector,
    μ::AbstractVector,
    diag_cov::AbstractVector,
    state,
) where {T}
    tot = Zygote.@ignore(sum(negbin_logconst(y, l.r))) - log(2.0) * sum(y .+ l.r)
    tot += 0.5 * dot(μ, (y .- l.r)) - 0.5 * dot(state.θ, μ) - 0.5 * dot(state.θ, diag_cov)
    return tot
end

function PolyaGammaKL(l::NegBinomialLikelihood, y::AbstractVector, state)
    return PolyaGammaKL(y .+ l.r, state.c, state.θ)
end
