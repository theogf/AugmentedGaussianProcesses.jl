"""
    LogisticSoftMaxLikelihood(num_class::Int)

## Arguments
- `num_class::Int` : Total number of classes

---
The multiclass likelihood with a logistic-softmax mapping: :
```math
p(y=i|{fₖ}₁ᴷ) = σ(fᵢ)/∑ₖ σ(fₖ)
```
where `σ` is the logistic function.
This likelihood has the same properties as [softmax](https://en.wikipedia.org/wiki/Softmax_function).
---

For the analytical version, the likelihood is augmented multiple times.
More details can be found in the paper [Multi-Class Gaussian Process Classification Made Conjugate: Efficient Inference via Data Augmentation](https://arxiv.org/abs/1905.09670)
"""
mutable struct LogisticSoftMaxLikelihood{T<:Real,A<:AbstractVector{T}} <:
               MultiClassLikelihood{T}
    nClasses::Int
    class_mapping::Vector{Any} # Classes labels mapping
    ind_mapping::Dict{Any,Int} # Mapping from label to index
    Y::Vector{BitVector} #Mapping from instances to classes (one hot encoding)
    y_class::Vector{Int} # GP Index for each sample
    c::Vector{A} # Second moment of fₖ
    α::A # First variational parameter of Gamma distribution
    β::A # Second variational parameter of Gamma distribution
    θ::Vector{A} # Variational parameter of Polya-Gamma distribution
    γ::Vector{A} # Variational parameter of Poisson distribution
    function LogisticSoftMaxLikelihood{T}(nClasses::Int) where {T<:Real}
        return new{T,Vector{T}}(nClasses)
    end
    function LogisticSoftMaxLikelihood{T}(
        nClasses::Int, labels::AbstractVector, ind_mapping::Dict
    ) where {T<:Real}
        return new{T,Vector{T}}(nClasses, labels, ind_mapping)
    end
    function LogisticSoftMaxLikelihood{T}(
        nClasses::Int,
        Y::AbstractVector{<:BitVector},
        class_mapping::AbstractVector,
        ind_mapping::Dict{<:Any,<:Int},
        y_class::AbstractVector{<:Int},
    ) where {T<:Real}
        return new{T,Vector{T}}(nClasses, class_mapping, ind_mapping, Y, y_class)
    end
    function LogisticSoftMaxLikelihood{T}(
        nClasses,
        Y::AbstractVector{<:BitVector},
        class_mapping::AbstractVector,
        ind_mapping::Dict{<:Any,<:Int},
        y_class::AbstractVector{<:Int},
        c::AbstractVector{A},
        α::A,
        β::A,
        θ::AbstractVector{A},
        γ::AbstractVector{A},
    ) where {T<:Real,A<:AbstractVector{T}}
        return new{T,A}(nClasses, class_mapping, ind_mapping, Y, y_class, c, α, β, θ, γ)
    end
end

LogisticSoftMaxLikelihood(nClasses::Int) = LogisticSoftMaxLikelihood{Float64}(nClasses)
function LogisticSoftMaxLikelihood(ylabels::AbstractVector)
    return LogisticSoftMaxLikelihood{Float64}(
        length(ylabels), ylabels, Dict(value => key for (key, value) in enumerate(ylabels))
    )
end

function implemented(
    ::LogisticSoftMaxLikelihood, ::Union{<:AnalyticVI,<:MCIntegrationVI,<:GibbsSampling}
)
    return true
end

function logisticsoftmax(f::AbstractVector{<:Real})
    return normalize(logistic.(f), 1)
end

function logisticsoftmax(f::AbstractVector{<:Real}, i::Integer)
    return logisticsoftmax(f)[i]
end

function (::LogisticSoftMaxLikelihood)(f::AbstractVector)
    return logisticsoftmax(f)
end

function (::LogisticSoftMaxLikelihood)(y::Integer, f::AbstractVector)
    return logisticsoftmax(f)[y]
end

function Base.show(io::IO, model::LogisticSoftMaxLikelihood{T}) where {T}
    return print(io, "Logistic-Softmax Likelihood")
end

function init_likelihood(
    l::LogisticSoftMaxLikelihood{T},
    inference::AbstractInference{T},
    nLatent::Integer,
    nSamplesUsed::Integer,
) where {T}
    if inference isa AnalyticVI || inference isa GibbsSampling
        c = [ones(T, nSamplesUsed) for i in 1:nLatent]
        α = nLatent * ones(T, nSamplesUsed)
        β = nLatent * ones(T, nSamplesUsed)
        θ = [abs.(rand(T, nSamplesUsed)) * 2 for i in 1:nLatent]
        γ = [abs.(rand(T, nSamplesUsed)) for i in 1:nLatent]
        LogisticSoftMaxLikelihood{T}(
            num_class(l), l.Y, l.class_mapping, l.ind_mapping, l.y_class, c, α, β, θ, γ
        )
    else
        return l
    end
end

## Local Updates ##
function local_updates!(
    l::LogisticSoftMaxLikelihood,
    y,
    μ::NTuple{N,<:AbstractVector},
    Σ::NTuple{N,<:AbstractVector},
) where {N}
    @. l.c = broadcast((Σ, μ) -> sqrt.(Σ + abs2.(μ)), Σ, μ)
    for _ in 1:2
        broadcast!(
            (β, c, μ, ψα) -> 0.5 / β * exp.(ψα) .* safe_expcosh.(-0.5 * μ, 0.5 * c),
            l.γ,
            Ref(l.β),
            l.c,
            μ,
            Ref(digamma.(l.α)),
        )
        l.α .= 1.0 .+ (l.γ...)
    end
    broadcast!(
        (y, γ, c) -> 0.5 * (y + γ) ./ c .* tanh.(0.5 .* c),
        l.θ, # target
        y, # argument 1
        l.γ, # argument 2
        l.c, # argument 3
    )
    return nothing
end

function sample_local!(l::LogisticSoftMaxLikelihood{T}, y::AbstractVector, f) where {T}
    broadcast!(f -> rand.(Poisson.(0.5 * l.α .* safe_expcosh.(-0.5 * f, 0.5 * f))), l.γ, f)
    l.α .= rand.(Gamma.(one(T) .+ (l.γ...), 1.0 ./ l.β))
    set_ω!(l, broadcast((y, γ, f) -> rand.(PolyaGamma.(y .+ Int.(γ), abs.(f))), y, l.γ, f))
    return nothing
end

## Global Gradient Section ##

@inline function ∇E_μ(l::LogisticSoftMaxLikelihood, ::AOptimizer, y::AbstractVector)
    return 0.5 .* (y .- l.γ)
end
@inline ∇E_Σ(l::LogisticSoftMaxLikelihood, ::AOptimizer, ::AbstractVector) = 0.5 .* l.θ

## ELBO Section ##
function expec_loglikelihood(
    l::LogisticSoftMaxLikelihood{T}, ::AnalyticVI, y, μ, Σ
) where {T}
    tot = -length(y) * logtwo
    tot += -sum(sum(l.γ .+ y)) * logtwo
    tot += 0.5 * sum(zip(l.θ, l.γ, y, μ, Σ)) do (θ, γ, y, μ, Σ)
        dot(μ, (y - γ)) - dot(θ, abs2.(μ)) - dot(θ, Σ)
    end
    return tot
end

function AugmentedKL(l::LogisticSoftMaxLikelihood, y::AbstractVector)
    return PolyaGammaKL(l, y) + PoissonKL(l) + GammaEntropy(l)
end

function PolyaGammaKL(l::LogisticSoftMaxLikelihood, y)
    return sum(broadcast(PolyaGammaKL, y .+ l.γ, l.c, l.θ))
end

function PoissonKL(l::LogisticSoftMaxLikelihood)
    return sum(broadcast(PoissonKL, l.γ, Ref(l.α ./ l.β), Ref(digamma.(l.α) .- log.(l.β))))
end

##  Compute the equivalent of KL divergence between an improper prior p(λ) (``1_{[0,\\infty]}``) and a variational Gamma distribution ##
function GammaEntropy(l::LogisticSoftMaxLikelihood)
    return -sum(l.α) + sum(log, first(l.β)) - sum(x -> first(logabsgamma(x)), l.α) -
           dot(1.0 .- l.α, digamma.(l.α))
end

## Numerical Gradient Section ##

function grad_samples(
    model::AbstractGPModel{T,<:LogisticSoftMaxLikelihood,<:NumericalVI},
    samples::AbstractMatrix{T},
    index::Int,
) where {T}
    class = model.likelihood.y_class[index]::Int
    grad_μ = zeros(T, nLatent(model))
    grad_Σ = zeros(T, nLatent(model))
    g_μ = similar(grad_μ)
    nSamples = size(samples, 1)
    @views @inbounds for i in 1:nSamples
        σ = logistic.(samples[i, :])
        samples[i, :] .= logisticsoftmax(samples[i, :])
        s = samples[i, class]
        g_μ .= grad_logisticsoftmax(samples[i, :], σ, class) / s
        grad_μ += g_μ
        grad_Σ += diaghessian_logisticsoftmax(samples[i, :], σ, class) / s - abs2.(g_μ)
    end
    for k in 1:nLatent(model)
        get_opt(inference(model), k).ν[index] = -grad_μ[k] / nSamples
        get_opt(inference(model), k).λ[index] = grad_Σ[k] / nSamples
    end
end

function log_like_samples(
    model::AbstractGPModel{T,<:LogisticSoftMaxLikelihood},
    samples::AbstractMatrix,
    index::Integer,
) where {T}
    class = model.likelihood.y_class[index]
    nSamples = size(samples, 1)
    loglike = zero(T)
    for i in 1:nSamples
        σ = logistic.(samples[i, :])
        loglike += log(σ[class]) - log(sum(σ))
    end
    return loglike / nSamples
end

function grad_logisticsoftmax(
    s::AbstractVector{T}, σ::AbstractVector{T}, i::Integer
) where {T<:Real}
    return s[i] * (δ.(T, i, eachindex(σ)) .- s) .* (1.0 .- σ)
end

function diaghessian_logisticsoftmax(
    s::AbstractVector{T}, σ::AbstractVector{T}, i::Integer
) where {T<:Real}
    return s[i] * (1.0 .- σ) .* (
        abs2.(δ.(T, i, eachindex(σ)) - s) .* (1.0 .- σ) - s .* (1.0 .- s) .* (1.0 .- σ) -
        σ .* (δ.(T, i, eachindex(σ)) - s)
    )
end

function hessian_logisticsoftmax(
    s::AbstractVector{T}, σ::AbstractVector{T}, i::Integer
) where {T<:Real}
    m = length(s)
    hessian = zeros(T, m, m)
    @inbounds for j in 1:m
        for k in 1:m
            hessian[j, k] =
                (1 - σ[j]) *
                s[i] *
                (
                    (δ(T, i, k) - s[k]) * (1.0 - σ[k]) * (δ(T, i, j) - s[j]) -
                    s[j] * (δ(T, j, k) - s[k]) * (1.0 - σ[k]) -
                    δ(T, k, j) * σ[j] * (δ(T, i, j) - s[j])
                )
        end
    end
    return hessian
end
