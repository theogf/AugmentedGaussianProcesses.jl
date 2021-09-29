@doc raw"""
    SoftMaxLikelihood(num_class::Int) -> MultiClassLikelihood

## Arguments
- `num_class::Int` : Total number of classes

    SoftMaxLikelihood(labels::AbstractVector) -> MultiClassLikelihood

## Arguments
- `labels::AbstractVector` : List of classes labels

Multiclass likelihood with [Softmax transformation](https://en.wikipedia.org/wiki/Softmax_function):

```math
p(y=i|\{f_k\}_{k=1}^K) = \frac{\exp(f_i)}{\sum_{k=1}^K\exp(f_k)}
```

There is no possible augmentation for this likelihood
"""
SoftMaxLikelihood(x) = MultiClassLikelihood(SoftMaxLink(), x)

implemented(::MultiClassLikelihood{<:SoftMaxLink}, ::MCIntegrationVI) = true

Base.show(io::IO, ::SoftMaxLink) = print(io, "SoftMax Link")

function grad_samples!(
    model::AbstractGPModel{T,<:MultiClassLikelihood{<:SoftMaxLink}},
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
    ::AbstractGPModel{T,<:MultiClassLikelihood{<:SoftMaxLink}},
    samples::AbstractMatrix,
    y::BitVector,
) where {T}
    num_sample = size(samples, 1)
    return mapslices(logsumexp, samples; dims=2) / num_sample
end

function grad_softmax(s::AbstractVector{<:Real}, y)
    return (y - s) * s[y][1]
end

function diaghessian_softmax(s::AbstractVector{<:Real}, y)
    return s[y][1] * (abs2.(y - s) - s .* (1 .- s))
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
