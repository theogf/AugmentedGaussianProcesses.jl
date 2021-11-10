## Solve the classical GP Regression ##
mutable struct Analytic{T<:Real} <: AbstractInference{T}
    ϵ::T # Convergence criteria
    n_iter::Int # Number of steps performed
    batchsize::Int
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    function Analytic{T}(ϵ::T) where {T}
        return new{T}(ϵ, 0, 0, false)
    end
end

"""
    Analytic(;ϵ::T=1e-5)

Analytic inference structure for solving the classical GP regression with Gaussian noise

## Keyword arguments
- `ϵ::Real` : convergence criteria (not used at the moment)
"""
Analytic

function Analytic(; ϵ::T=1e-5) where {T<:Real}
    return Analytic{T}(ϵ)
end

function Base.show(io::IO, ::Analytic)
    return print(io, "Analytic Inference")
end

function init_inference(
    i::Analytic{T}, nSamples::Integer, xview::TX, yview::TY
) where {T<:Real,TX,TY}
    return Analytic{T}(conv_crit(i), nSamples, xview, yview)
end

function analytic_updates(m::GP{T}, state, y) where {T}
    f = getf(m)
    l = likelihood(m)
    K = state.kernel_matrices.K
    f.post.Σ = K + only(l.σ²) * I
    f.post.α .= cov(f) \ (y - pr_mean(f, input(m.data)))
    if !isnothing(l.opt_noise)
        g = (norm(mean(f), 2) - tr(inv(cov(f)))) / 2
        state_σ², Δlogσ² = Optimisers.apply(
            l.opt_noise, state.local_vars.state_σ², l.σ², g .* l.σ²
        )
        local_vars = merge(state.local_vars, (; state_σ²))
        state = merge(state, (; local_vars))
        l.σ² .= exp.(log.(l.σ²) .+ Δlogσ²)
    end
    return state
end

ρ(::Analytic{T}) where {T} = one(T)

is_stochastic(::Analytic) = false
