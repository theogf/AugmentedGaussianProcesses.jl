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

Note that this likelihood follows the Wikipedia definition and not the Distributions.jl one.
"""
struct NegBinomialLikelihood{L,Tr} <: AbstractLikelihood
    invlink::L
    r::Tr
end

NegBinomialLikelihood(r) = NegBinomialLikelihood(LogisticLink(), r)

(l::NegBinomialLikelihood)(f::Real) = NegativeBinomial(l.r, l.invlink(-f))

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
    T = eltype(μ)
    pred = zeros(T, N)
    sig_pred = zeros(T, N)
    function negbin_mean(x)
        p = l.invlink(x)
        return p * l.r / (1 - p)
    end
    for i in 1:N
        x = pred_nodes .* sqrt(max(σ²[i], zero(T))) .+ μ[i]
        pred[i] = dot(pred_weights, negbin_mean.(x))
        sig_pred[i] = dot(pred_weights, negbin_mean.(x) .^ 2) - pred[i]^2
    end
    return pred, sig_pred
end

## Local Updates ##
function init_local_vars(::NegBinomialLikelihood, batchsize::Int, T::DataType=Float64)
    return (; c=rand(T, batchsize), θ=zeros(T, batchsize))
end

function local_updates!(
    local_vars,
    l::NegBinomialLikelihood,
    y::AbstractVector,
    μ::AbstractVector,
    diagΣ::AbstractVector,
)
    map!(sqrt_expec_square, local_vars.c, μ, diagΣ) # √E[f^2]
    map!(local_vars.θ, local_vars.c, y) do c, y
        (l.r + y) * tanh(c / 2) / c
    end # E[ω]
    return local_vars
end

function sample_local!(
    local_vars, l::NegBinomialLikelihood, y::AbstractVector, f::AbstractVector
)
    map!(local_vars.θ, f, y) do f, y
        rand(PolyaGamma(y + Int(l.r), abs(f)))
    end
    return local_vars
end

## Global Updates ##

@inline function ∇E_μ(l::NegBinomialLikelihood, ::AOptimizer, y::AbstractVector, state)
    return ((y .- l.r) / 2,)
end
@inline function ∇E_Σ(::NegBinomialLikelihood, ::AOptimizer, y::AbstractVector, state)
    return (state.θ / 2,)
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
    tot = ChainRulesCore.@ignore_derivatives(sum(negbin_logconst(y, l.r))) - log(2.0) * sum(y .+ l.r)
    tot += dot(μ, (y .- l.r)) / 2 - dot(state.θ, μ) / 2 - dot(state.θ, diag_cov) / 2
    return tot
end

function PolyaGammaKL(l::NegBinomialLikelihood, state, y)
    return PolyaGammaKL(y .+ l.r, state.c, state.θ)
end
