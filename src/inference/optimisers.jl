struct RobbinsMonro{T}
    κ::T
    τ::T
end

function RobbinsMonro(κ::Real=0.51, τ::Real=1)
    0.5 < κ <= 1 || error("κ should be in the interval (0.5,1]")
    τ > 0 || error("τ should be positive")
    return RobbinsMonro{promote_type(typeof(κ), typeof(τ))}(κ, τ)
end

Optimisers.init(::RobbinsMonro, ::Any) = 1

function Optimisers.apply(o::RobbinsMonro, st, x, Δ)
    κ = o.κ
    τ = o.τ
    n = st
    return (n + 1), Δ * 1 / (τ + n)^κ
end

"""
    ALRSVI(n_mc=10, \rho)

Adaptive Learning Rate for Stochastic Variational Inference
"""
struct ALRSVI{T}
    n_mc::Int # Number of initial MC steps
    ρ::T # Initial learning rate
end

function Optimisers.init(opt::ALRSVI{T}, x::AbstractArray) where {T}
    i = 1
    g = zero(x)
    h = norm(g)
    τ = zero(T)
    return (; i, g, h, ρ=opt.ρ, τ)
end

function apply(opt::ALRSVI, state, x::AbstractArray, Δx::AbstractArray)
    if state.i <= opt.n_mc
        g = state.g + Δx
        h = state.h + norm(Δx)
        ρ = opt.ρ
        τ = state.τ
        if state.i == opt.τ
            g = g / τ
            h = h / τ
            ρ = sum(abs2, g) / h
        end
    else
        g = (1 - 1 / opt.τ) * state.g + 1 / opt.τ * Δx
        h = (1 - 1 / opt.τ) * state.h + 1 / opt.τ * sum(abs2, Δx)
        ρ = sum(abs2, g) / h
        τ = state.τ * (1 - ρ) + 1.0
    end

    return (; i=(i + 1), g, h, ρ, τ), ρ * Δx
end
