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
struct NegBinomialLikelihood{L,Tr} <: EventLikelihood
    invlink::L
    r::Tr
end

NegBinomialLikelihood(r) = NegBinomialLikelihood(LogisticLink(), r)

(l::NegBinomialLikelihood)(f::Real) = NegativeBinomial(l.r, l.invlink(f))

implemented(::NegBinomialLikelihood, ::Union{<:AnalyticVI,<:GibbsSampling}) = true

function (l::NegBinomialLikelihood)(y::Real, f::Real)
    return pdf(l(f), y)
end

function Distributions.loglikelihood(l::NegBinomialLikelihood, y::Real, f::Real)
    return logpdf(l(f), y)
end

function Base.show(io::IO, l::NegBinomialLikelihood)
    return print(io, "Negative Binomial Likelihood (r = $(l.r))")
end

function compute_proba(
    l::NegBinomialLikelihood, μ::AbstractVector{<:Real}, σ²::AbstractVector{<:Real}
)
    N = length(μ)
    pred = zeros(T, N)
    sig_pred = zeros(T, N)
    for i in 1:N
        x = pred_nodes .* sqrt(max(σ²[i], zero(T))) .+ μ[i]
        pred[i] = dot(pred_weights, l.invlink.(x))
        sig_pred[i] = dot(pred_weights, l.invlink.(x) .^ 2) - pred[i]^2
    end
    return pred, sig_pred
end

## Local Updates ##
function init_local_vars(state, ::NegBinomialLikelihood, batchsize::Int, T::DataType=Float64)
    return merge(state, (; local_vars=(; c=rand(T, batchsize), θ=zeros(T, batchsize))))
end

function local_updates!(
    local_vars,
    l::NegBinomialLikelihood,
    y::AbstractVector,
    μ::AbstractVector,
    Σ::AbstractVector,
)
    @. local_vars.c = sqrt(abs2(μ) + Σ)
    @. local_vars.θ = (l.r + y) / local_vars.c * tanh(0.5 * local_vars.c)
    return local_vars
end

function sample_local!(
    local_vars, l::NegBinomialLikelihood, y::AbstractVector, f::AbstractVector
)
    local_vars.θ .= rand.(PolyaGamma.(y .+ Int(l.r), abs.(f)))
    return local_vars
end

## Global Updates ##

@inline function ∇E_μ(
    l::NegBinomialLikelihood, ::AOptimizer, y::AbstractVector, state
)
    return (0.5 * (y .- l.r),)
end
@inline function ∇E_Σ(
    ::NegBinomialLikelihood, ::AOptimizer, y::AbstractVector, state
)
    return (0.5 .* state.θ,)
end

## ELBO Section ##

AugmentedKL(l::NegBinomialLikelihood, state, y) = PolyaGammaKL(l, state, y)

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
    l::NegBinomialLikelihood,
    ::AnalyticVI,
    y::AbstractVector,
    μ::AbstractVector,
    diag_cov::AbstractVector,
    state,
)
    tot = Zygote.@ignore(sum(negbin_logconst(y, l.r))) - log(2.0) * sum(y .+ l.r)
    tot += 0.5 * dot(μ, (y .- l.r)) - 0.5 * dot(state.θ, μ) - 0.5 * dot(state.θ, diag_cov)
    return tot
end

function PolyaGammaKL(l::NegBinomialLikelihood, state, y)
    return PolyaGammaKL(y .+ l.r, state.c, state.θ)
end
