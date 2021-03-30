"""
```julia
    SoftMaxLikelihood()
```
Multiclass likelihood with [Softmax transformation](https://en.wikipedia.org/wiki/Softmax_function):

```math
p(y=i|{fₖ}) = exp(fᵢ)/ ∑ₖexp(fₖ)
```

There is no possible augmentation for this likelihood
"""
mutable struct SoftMaxLikelihood{T<:Real,A<:AbstractVector{T}} <: MultiClassLikelihood{T}
    nClasses::Int
    class_mapping::Vector{Any} # Classes labels mapping
    ind_mapping::Dict{Any,Int} # Mapping from label to index
    Y::Vector{BitVector} #Mapping from instances to classes
    y_class::Vector{Int64} #GP Index for each sample
    function SoftMaxLikelihood{T}(nClasses::Int) where {T<:Real}
        return new{T,Vector{T}}(nClasses)
    end
    function SoftMaxLikelihood{T}(
        nClasses::Int, labels::AbstractVector, ind_mapping::Dict
    ) where {T<:Real}
        return new{T,Vector{T}}(nClasses, labels, ind_mapping)
    end
    function SoftMaxLikelihood{T}(
        Y::AbstractVector{<:BitVector},
        class_mapping::AbstractVector,
        ind_mapping::Dict{<:Any,<:Int},
        y_class::AbstractVector{<:Int},
    ) where {T<:Real}
        return new{T,Vector{T}}(Y, class_mapping, ind_mapping, y_class)
    end
end

SoftMaxLikelihood(nClasses::Int) = SoftMaxLikelihood{Float64}(nClasses)
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

function Base.show(io::IO, model::SoftMaxLikelihood{T}) where {T}
    return print(io, "Softmax likelihood")
end

function init_likelihood(
    likelihood::SoftMaxLikelihood{T},
    inference::AbstractInference{T},
    nLatent::Int,
    nSamplesUsed::Int,
) where {T}
    if inference isa GibbsSampling
        θ = [abs.(rand(T, nSamplesUsed)) * 2 for i in 1:nLatent]
        LogisticSoftMaxLikelihood{T}(
            likelihood.Y,
            likelihood.class_mapping,
            likelihood.ind_mapping,
            likelihood.y_class,
            θ,
        )
    else
        return likelihood
    end
end

function sample_local!(model::VGP{T,<:SoftMaxLikelihood,<:GibbsSampling}) where {T}
    model.likelihood.θ .= broadcast(
        (y::BitVector, γ::AbstractVector{<:Real}, μ::AbstractVector{<:Real}, i::Int64) ->
            rand.(PolyaGamma.(1.0, μ - logsumexp())),
        model.likelihood.Y,
        model.likelihood.γ,
        model.μ,
        1:(model.nLatent),
    )
    return nothing #TODO FINISH AT SOME POINT
end

function grad_samples(
    model::AbstractGP{T,<:SoftMaxLikelihood}, samples::AbstractMatrix{T}, index::Integer
) where {T}
    class = model.likelihood.y_class[index]
    grad_μ = zeros(T, nLatent(model))
    grad_Σ = zeros(T, nLatent(model))
    nSamples = size(samples, 1)
    samples .= mapslices(StatsFuns.softmax, samples; dims=2)
    t = 0.0
    @inbounds for i in 1:nSamples
        s = samples[i, class]
        @views g_μ = grad_softmax(samples[i, :], class) / s
        grad_μ += g_μ
        @views h = diaghessian_softmax(samples[i, :], class) / s
        grad_Σ += h - abs2.(g_μ)
    end
    for k in 1:nLatent(model)
        get_opt(inference(model), k).ν[index] = -grad_μ[k] / nSamples
        get_opt(inference(model), k).λ[index] = grad_Σ[k] / nSamples
    end
end

function log_like_samples(
    model::AbstractGP{T,<:SoftMaxLikelihood}, samples::AbstractMatrix, index::Integer
) where {T}
    class = model.likelihood.y_class[index]
    nSamples = size(samples, 1)
    return loglike = mapslices(logsumexp, samples; dims=2) / nSamples
end

function grad_softmax(s::AbstractVector{<:Real}, i::Integer)
    return (δ.(i, eachindex(s)) - s) * s[i]
end

function diaghessian_softmax(s::AbstractVector{<:Real}, i::Integer)
    return s[i] * (abs2.(δ.(i, eachindex(s)) - s) - s .* (1.0 .- s))
end

function hessian_softmax(s::AbstractVector{<:Real}, i::Integer)
    m = length(s)
    hessian = zeros(m, m)
    for j in 1:m
        for k in 1:m
            hessian[j, k] =
                s[i] * ((δ(i, k) - s[k]) * (δ(i, j) - s[j]) - s[j] * (δ(j, k) - s[k]))
        end
    end
    return hessian
end
