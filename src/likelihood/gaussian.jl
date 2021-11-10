@doc raw"""
    GaussianLikelihood(σ²::T=1e-3) # σ² is the variance of the noise

Gaussian noise :
```math
    p(y|f) = N(y|f,\sigma^2)
```
There is no augmentation needed for this likelihood which is already conjugate to a Gaussian prior.
"""
struct GaussianLikelihood{T<:Real,O} <: RegressionLikelihood
    σ²::Vector{T}
    opt_noise::O
    function GaussianLikelihood{T}(σ²::T, opt_noise) where {T<:Real}
        return new{T,typeof(opt_noise)}([σ²], opt_noise)
    end
end

function GaussianLikelihood(σ²::T=1e-3; opt_noise=false) where {T<:Real}
    if isa(opt_noise, Bool)
        opt_noise = opt_noise ? ADAM(0.05) : nothing
    end
    return GaussianLikelihood{T}(σ², opt_noise)
end

implemented(::GaussianLikelihood, ::Union{<:AnalyticVI,<:Analytic}) = true

function (l::GaussianLikelihood)(y::Real, f::Real)
    return pdf(Normal(y, sqrt(noise(l))), f)
end

function Distributions.loglikelihood(l::GaussianLikelihood, y::Real, f::Real)
    return logpdf(Normal(y, sqrt(noise(l))), f)
end

noise(l::GaussianLikelihood) = only(l.σ²)

function Base.show(io::IO, l::GaussianLikelihood)
    return print(io, "Gaussian likelihood (σ² = $(noise(l)))")
end

function compute_proba(
    l::GaussianLikelihood, μ::AbstractVector{<:Real}, σ²::AbstractVector{<:Real}
)
    return μ, σ² .+ noise(l)
end

function init_local_vars(l::GaussianLikelihood, batchsize::Int, T::DataType=Float64)
    local_vars = (; θ=fill(inv(l.σ²[1]), batchsize))
    if !isnothing(l.opt_noise)
        state_σ² = Optimisers.init(l.opt_noise, l.σ²)
        local_vars = merge(local_vars, (; state_σ²))
    end
    return local_vars
end

function local_updates!(
    local_vars,
    l::GaussianLikelihood,
    y::AbstractVector,
    μ::AbstractVector,
    var_f::AbstractVector,
)
    if !isnothing(l.opt_noise)
        grad = ((sum(abs2, y - μ) + sum(var_f)) / noise(l) - length(y)) / 2
        gradlog, local_vars.state_σ² = Optimisers.apply!(
            l.opt_noise, local_vars.state_σ², l.σ², [grad]
        )
        l.σ² .= exp.(log.(l.σ²) + gradlog)
    end
    local_vars.θ .= inv(noise(l))
    return local_vars
end

@inline function ∇E_μ(l::GaussianLikelihood, ::AOptimizer, y::AbstractVector, state)
    return (y ./ noise(l),)
end

@inline function ∇E_Σ(::GaussianLikelihood, ::AOptimizer, ::AbstractVector, state)
    return (state.θ / 2,)
end

function expec_loglikelihood(
    l::GaussianLikelihood,
    ::AnalyticVI,
    y::AbstractVector,
    μ::AbstractVector,
    diagΣ::AbstractVector,
    state,
)
    return -(
        length(y) * (log(twoπ) + log(noise(l))) + (sum(abs2, y - μ) + sum(diagΣ)) / noise(l)
    ) / 2
end

AugmentedKL(::GaussianLikelihood, state, ::Any) = 0
