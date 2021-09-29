@doc raw"""
    LogisticSoftMaxLikelihood(num_class::Int) -> MultiClassLikelihood

## Arguments
- `num_class::Int` : Total number of classes

    LogisticSoftMaxLikelihood(labels::AbstractVector) -> MultiClassLikelihood

## Arguments
- `labels::AbstractVector` : List of classes labels

---

The multiclass likelihood with a logistic-softmax mapping: :
```math
p(y=i|\{f_k\}_{1}^{K}) = \frac{\sigma(f_i)}{\sum_{k=1}^k \sigma(f_k)}
```
where ``\sigma`` is the logistic function.
This likelihood has the same properties as [softmax](https://en.wikipedia.org/wiki/Softmax_function).
---

For the analytical version, the likelihood is augmented multiple times.
More details can be found in the paper [Multi-Class Gaussian Process Classification Made Conjugate: Efficient Inference via Data Augmentation](https://arxiv.org/abs/1905.09670).
"""
LogisticSoftMaxLikelihood(x) = MultiClassLikelihood(LogisticSoftMaxLink(), x)

struct LogisticSoftMaxLink <: AbstractLink end

function (::LogisticSoftMaxLink)(f::AbstractVector{<:Real})
    return normalize(logistic.(f), 1)
end

function implemented(
    ::MultiClassLikelihood{<:LogisticSoftMaxLink},
    ::Union{<:AnalyticVI,<:MCIntegrationVI,<:GibbsSampling},
)
    return true
end

Base.show(io::IO, ::LogisticSoftMaxLink) = print(io, "Logistic-SoftMax Link")

## Local Updates ##
function init_local_vars(
    l::MultiClassLikelihood{<:LogisticSoftMaxLink}, batchsize::Int, T::DataType=Float64
)
    num_class = n_class(l)
    c = [ones(T, batchsize) for _ in 1:num_class] # Second moment of fₖ
    α = num_class * ones(T, batchsize) # First variational parameter of Gamma distribution
    β = num_class * ones(T, batchsize) # Second variational parameter of Gamma distribution
    θ = [rand(T, batchsize) * 2 for _ in 1:num_class] # Variational parameter of Polya-Gamma distribution
    γ = [rand(T, batchsize) for _ in 1:num_class] # Variational parameter of Poisson distribution
    return (; c, α, β, θ, γ)
end

function local_updates!(
    local_vars,
    ::MultiClassLikelihood{<:LogisticSoftMaxLink},
    y,
    μ::NTuple{N,<:AbstractVector},
    Σ::NTuple{N,<:AbstractVector},
) where {N}
    @. local_vars.c = broadcast((Σ, μ) -> sqrt.(Σ + abs2.(μ)), Σ, μ)
    for _ in 1:2
        broadcast!(
            (β, c, μ, ψα) -> 0.5 / β * exp.(ψα) .* safe_expcosh.(-0.5 * μ, 0.5 * c),
            local_vars.γ,
            Ref(local_vars.β),
            local_vars.c,
            μ,
            Ref(digamma.(local_vars.α)),
        ) # Update γ
        local_vars.α .= 1 .+ (local_vars.γ...)
    end
    broadcast!(
        (y, γ, c) -> 0.5 * (y + γ) ./ c .* tanh.(0.5 .* c),
        local_vars.θ, # target
        eachcol(y), # argument 1
        local_vars.γ, # argument 2
        local_vars.c, # argument 3
    ) # update θ
    return local_vars
end

function sample_local!(local_vars, ::MultiClassLikelihood{<:LogisticSoftMaxLink}, y, f)
    broadcast!(
        f -> rand.(Poisson.(0.5 * local_vars.α .* safe_expcosh.(-0.5 * f, 0.5 * f))),
        local_vars.γ,
        f,
    )
    local_vars.α .= rand.(Gamma.(1 .+ (local_vars.γ...), inv.(local_vars.β)))
    local_vars.θ .= broadcast(
        (y, γ, f) -> rand.(PolyaGamma.(y .+ Int.(γ), abs.(f))), eachcol(y), local_vars.γ, f
    )
    return local_vars
end

## Global Gradient Section ##

@inline function ∇E_μ(::MultiClassLikelihood{<:LogisticSoftMaxLink}, ::AOptimizer, y, state)
    return 0.5 .* (eachcol(y) .- state.γ)
end
@inline function ∇E_Σ(::MultiClassLikelihood{<:LogisticSoftMaxLink}, ::AOptimizer, y, state)
    return 0.5 .* state.θ
end

## ELBO Section ##
function expec_loglikelihood(
    ::MultiClassLikelihood{<:LogisticSoftMaxLink}, ::AnalyticVI, y, μ, Σ, state
)
    tot = -length(y) * logtwo
    tot += -sum(sum(state.γ .+ eachcol(y))) * logtwo
    tot += 0.5 * sum(zip(state.θ, state.γ, eachcol(y), μ, Σ)) do (θ, γ, y, μ, Σ)
        dot(μ, (y - γ)) - dot(θ, abs2.(μ)) - dot(θ, Σ)
    end
    return tot
end

function AugmentedKL(l::MultiClassLikelihood{<:LogisticSoftMaxLink}, state, y)
    return PolyaGammaKL(l, state, y) + PoissonKL(l, state) + GammaEntropy(l, state)
end

function PolyaGammaKL(::MultiClassLikelihood, state, y)
    return sum(broadcast(PolyaGammaKL, eachcol(y) .+ state.γ, state.c, state.θ))
end

function PoissonKL(::MultiClassLikelihood, state)
    return sum(
        broadcast(
            PoissonKL,
            state.γ,
            Ref(state.α ./ state.β),
            Ref(digamma.(state.α) .- log.(state.β)),
        ),
    )
end

##  Compute the equivalent of KL divergence between an improper prior p(λ) (``1_{[0,\\infty]}``) and a variational Gamma distribution ##
function GammaEntropy(::MultiClassLikelihood, state)
    return -sum(state.α) + sum(log, first(state.β)) - sum(first ∘ logabsgamma, state.α) -
           dot(1 .- state.α, digamma.(state.α))
end

## Numerical Gradient Section ##

function grad_samples!(
    model::AbstractGPModel{T,<:MultiClassLikelihood{<:LogisticSoftMaxLink},<:NumericalVI},
    samples::AbstractMatrix{T},
    opt_state,
    y,
    index,
) where {T}
    grad_μ = zeros(T, n_latent(model))
    grad_Σ = zeros(T, n_latent(model))
    g_μ = similar(grad_μ)
    num_sample = size(samples, 1)
    @views @inbounds for i in 1:num_sample
        σ = logistic.(samples[i, :])
        samples[i, :] .= logisticsoftmax(samples[i, :])
        s = samples[i, y][1]
        g_μ .= grad_logisticsoftmax(samples[i, :], σ, y) / s
        grad_μ += g_μ
        grad_Σ += diaghessian_logisticsoftmax(samples[i, :], σ, y) / s - abs2.(g_μ)
    end
    for k in 1:n_latent(model)
        opt_state[k].ν[index] = -grad_μ[k] / num_sample
        opt_state[k].λ[index] = grad_Σ[k] / num_sample
    end
end

function log_like_samples(
    ::AbstractGPModel{T,<:MultiClassLikelihood{<:LogisticSoftMaxLink}},
    samples::AbstractMatrix,
    y,
) where {T}
    num_sample = size(samples, 1)
    loglike = zero(T)
    for i in 1:num_sample
        σ = logistic.(samples[i, :])
        loglike += log(σ[y][1]) - log(sum(σ))
    end
    return loglike / num_sample
end

function grad_logisticsoftmax(s::AbstractVector{T}, σ::AbstractVector{T}, y) where {T<:Real}
    return s[y][1] * (y .- s) .* (one(T) .- σ)
end

function diaghessian_logisticsoftmax(
    s::AbstractVector{T}, σ::AbstractVector{T}, y
) where {T<:Real}
    return s[y][1] * (one(T) .- σ) .*
           (abs2.(y - s) .* (1 .- σ) - s .* (1 .- s) .* (1 .- σ) - σ .* (y - s))
end

function hessian_logisticsoftmax(
    s::AbstractVector{T}, σ::AbstractVector{T}, y
) where {T<:Real}
    m = length(s)
    i = findfirst(y)
    hessian = zeros(T, m, m)
    @inbounds for j in 1:m
        for k in 1:m
            hessian[j, k] =
                (one(T) - σ[j]) *
                s[i] *
                (
                    (δ(T, i, k) - s[k]) * (one(T) - σ[k]) * (δ(T, i, j) - s[j]) -
                    s[j] * (δ(T, j, k) - s[k]) * (one(T) - σ[k]) -
                    δ(T, k, j) * σ[j] * (δ(T, i, j) - s[j])
                )
        end
    end
    return hessian
end
