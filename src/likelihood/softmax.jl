@doc raw"""
    SoftMaxLikelihood()

Multiclass likelihood with [Softmax transformation](https://en.wikipedia.org/wiki/Softmax_function):

```math
p(y=i|\{f_k\}_{k=1}^K) = \frac{\exp(f\_i)}{\sum_{k=1}^K\exp(f_k)}
```

There is no possible augmentation for this likelihood
"""
mutable struct SoftMaxLikelihood{T<:Real,A<:AbstractVector{T}} <: MultiClassLikelihood{T}
    nClasses::Int
    class_mapping::Vector{Any} # Classes labels mapping
    ind_mapping::Dict{Any,Int} # Mapping from label to index
end

SoftMaxLikelihood(n_class::Int) = SoftMaxLikelihood{Float64}(n_class)
function SoftMaxLikelihood(ylabels::AbstractVector)
    return SoftMaxLikelihood{Float64}(
        length(ylabels), ylabels, Dict(value => key for (key, value) in enumerate(ylabels))
    )
end

implemented(::SoftMaxLikelihood, ::MCIntegrationVI) = true

function (::SoftMaxLikelihood)(f::AbstractVector)
    return StatsFuns.softmax(f)
end

function (l::SoftMaxLikelihood)(y::Int, f::AbstractVector{<:Real})
    return StatsFuns.softmax(f)[y]
end

function Base.show(io::IO, ::SoftMaxLikelihood{T}) where {T}
    return print(io, "Softmax likelihood")
end

function sample_local!(local_vars, model::VGP{T,<:SoftMaxLikelihood,<:GibbsSampling}) where {T}
    model.likelihood.θ .= broadcast(
        (y::BitVector, γ::AbstractVector{<:Real}, μ::AbstractVector{<:Real}, i::Int64) ->
            rand.(PolyaGamma.(1.0, μ - logsumexp(μ))),
        model.likelihood.Y,
        model.likelihood.γ,
        model.μ,
        1:(model.nLatent),
    )
    return local_vars #TODO FINISH AT SOME POINT
end

function grad_samples(
    model::AbstractGPModel{T,<:SoftMaxLikelihood},
    samples::AbstractMatrix{T},
    opt_state,
    y,
    index,
) where {T}
    grad_μ = zeros(T, n_latent(model))
    grad_Σ = zeros(T, n_latent(model))
    num_sample = size(samples, 1)
    samples .= mapslices(StatsFuns.softmax, samples; dims=2)
    @inbounds for i in 1:num_sample
        s = samples[i, y][1]
        @views g_μ = grad_softmax(samples[i, :], y) / s
        grad_μ += g_μ
        @views h = diaghessian_softmax(samples[i, :], y) / s
        grad_Σ += h - abs2.(g_μ)
    end
    for k in 1:n_latent(model)
        opt_state[k].ν[index] = -grad_μ[k] / num_sample
        opt_state[k].λ[index] = grad_Σ[k] / num_sample
    end
end

function log_like_samples(
    ::AbstractGPModel{T,<:SoftMaxLikelihood}, samples::AbstractMatrix, y::BitVector
) where {T}
    num_sample = size(samples, 1)
    return mapslices(logsumexp, samples; dims=2) / num_sample
end

function grad_softmax(s::AbstractVector{<:Real}, y)
    return (y - s) * s[i]
end

function diaghessian_softmax(s::AbstractVector{<:Real}, y)
    return s[i] * (abs2.(y - s) - s .* (1 .- s))
end

function hessian_softmax(s::AbstractVector{T}, y) where {T}
    m = length(s)
    i = findfirst(y)
    hessian = zeros(T, m, m)
    for j in 1:m
        for k in 1:m
            hessian[j, k] =
                s[i] * ((δ(i, k) - s[k]) * (δ(i, j) - s[j]) - s[j] * (δ(j, k) - s[k]))
        end
    end
    return hessian
end
