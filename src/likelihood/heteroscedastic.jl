@doc raw"""
    HeteroscedasticLikelihood(λ::T=1.0)->HeteroscedasticGaussianLikelihood

## Arguments
- `λ::Real` : The maximum precision possible (this is optimized during training)

---

Gaussian with heteroscedastic noise given by another gp:
```math
    p(y|f,g) = \mathcal{N}(y|f,(\lambda \sigma(g))^{-1})
```
Where ``\sigma`` is the logistic function

The augmentation is not trivial and will be described in a future paper
"""
HeteroscedasticLikelihood(λ::Real) = HeteroscedasticGaussianLikelihood(InvScaledLogistic(λ))

struct InvScaledLogistic{T} <: AbstractLink
    λ::Vector{T}
end

InvScaledLogistic(λ::Real) = InvScaledLogistic([λ])

(l::InvScaledLogistic)(f::Real) = inv(l.λ[1] * logistic(f))

function implemented(
    ::HeteroscedasticGaussianLikelihood{<:InvScaledLogistic},
    ::Union{<:AnalyticVI,<:GibbsSampling},
)
    return true
end

function (l::HeteroscedasticGaussianLikelihood)(f, y::Real)
    return pdf(l(f), y)
end

function Distributions.loglikelihood(
    l::HeteroscedasticGaussianLikelihood, f::AbstractVector, y
)
    return logpdf(l(f), y)
end

function Base.show(io::IO, ::HeteroscedasticGaussianLikelihood)
    return print(io, "Gaussian likelihood with heteroscedastic noise")
end

n_latent(::HeteroscedasticGaussianLikelihood) = 2

function init_local_vars(
    ::HeteroscedasticGaussianLikelihood{<:InvScaledLogistic},
    batchsize::Int,
    T::DataType=Float64,
)
    return (;
        c=ones(T, batchsize), # Sqrt of expectation of g^2
        ϕ=ones(T, batchsize), # Expectation of (y - f)^2 / 2
        γ=ones(T, batchsize), # Expectation of q(n)
        θ=ones(T, batchsize), # Expectation of q(ω|n)
        σg=ones(T, batchsize), # Approximation of the expectation of σ(-g)
    )
end

function compute_proba(
    l::HeteroscedasticGaussianLikelihood,
    μ::AbstractVector{<:AbstractVector},
    Σ::AbstractVector{<:AbstractVector},
)
    return μ[1], Σ[1] .+ l.invlink.(μ[2])
end

function local_updates!(
    local_vars,
    l::HeteroscedasticGaussianLikelihood,
    y::AbstractVector,
    μ::NTuple{2,<:AbstractVector},
    diagΣ::NTuple{2,<:AbstractVector},
)
    # gp[1] is f and gp[2] is g (for approximating the noise)
    λ = only(l.invlink.λ)
    map!(local_vars.ϕ, μ[1], diagΣ[1], y) do μ, σ², y
        (abs2(μ - y) + σ²) / 2 # E[(f-y)^2/2]
    end
    map!(sqrt_expec_square, local_vars.c, μ[2], diagΣ[2]) # √E[g^2]
    map!(local_vars.σg, μ[2], local_vars.c) do μ, c
        safe_expcosh(-μ / 2, c / 2) / 2 # ≈ E[σ(-g)]
    end
    map!(local_vars.γ, local_vars.ϕ, local_vars.σg) do ϕ, σg
        λ * ϕ * σg
    end # E[n]
    map!(local_vars.θ, local_vars.γ, local_vars.c) do γ, c
        (1//2 + γ) * tanh(c / 2) / (2c)
    end # E[ω]
    l.invlink.λ .= max(length(y) / (2 * dot(local_vars.ϕ, 1 .- local_vars.σg)), λ)
    return local_vars
end

function sample_local!(
    local_vars, l::HeteroscedasticGaussianLikelihood, y, f::NTuple{2,<:AbstractVector}
)
    λ = only(l.invlink.λ)
    map!(local_vars.γ, f[1], f[2], y) do f, g, y
        rand(Poisson(λ * logistic(g) * abs2(f - y) / 2))
    end # Update of n
    map!(local_vars.θ, local_vars.γ, f[2]) do n, g
        rand(PolyaGamma(n + 1//2, abs(g)))
    end # update of ω
    return local_vars
end

@inline function ∇E_μ(
    l::HeteroscedasticGaussianLikelihood{<:InvScaledLogistic},
    ::AOptimizer,
    y::AbstractVector,
    state,
)
    return (y .* only(l.invlink.λ) .* state.σg / 2, (1//2 .- state.γ) / 2)
end

@inline function ∇E_Σ(
    l::HeteroscedasticGaussianLikelihood{<:InvScaledLogistic},
    ::AOptimizer,
    ::AbstractVector,
    state,
)
    return (only(l.invlink.λ) * state.σg / 2, state.θ / 2)
end

function compute_proba(
    l::HeteroscedasticGaussianLikelihood,
    μs::Tuple{<:AbstractVector,<:AbstractVector},
    σs::Tuple{<:AbstractVector,<:AbstractVector},
) where {T<:Real}
    return μs[1], σs[1] + l.invlink.(μs[2])
end

function predict_y(
    ::HeteroscedasticGaussianLikelihood, μs::Tuple{<:AbstractVector,<:AbstractVector}
)
    return first(μs) # For predict_y the variance is ignored
end

function expec_loglikelihood(
    l::HeteroscedasticGaussianLikelihood{<:InvScaledLogistic},
    ::AnalyticVI,
    y::AbstractVector,
    μ,
    diag_cov,
    state,
)
    tot = length(y) * (log(l.invlink.λ[1]) / 2 - log(2 * sqrt(twoπ)))
    tot +=
        (
            dot(μ[2], (1//2 .- state.γ)) - dot(abs2.(μ[2]), state.θ) -
            dot(diag_cov[2], state.θ)
        ) / 2
    tot -= PoissonKL(l, y, μ[1], diag_cov[1], state)
    return tot
end

function AugmentedKL(
    l::HeteroscedasticGaussianLikelihood{<:InvScaledLogistic}, state, ::Any
)
    return PolyaGammaKL(l, state)
end

function PoissonKL(
    l::HeteroscedasticGaussianLikelihood{<:InvScaledLogistic},
    y::AbstractVector,
    μ::AbstractVector,
    Σ::AbstractVector,
    state,
)
    λ = only(l.invlink.λ)
    return PoissonKL(state.γ, λ * (abs2.(y - μ) + Σ) / 2, log.(λ * (abs2.(μ - y) + Σ) / 2))
end

function PolyaGammaKL(::HeteroscedasticGaussianLikelihood{<:InvScaledLogistic}, state)
    return PolyaGammaKL(1//2 .+ state.γ, state.c, state.θ)
end
