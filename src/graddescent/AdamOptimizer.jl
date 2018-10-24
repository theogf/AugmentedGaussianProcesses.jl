mutable struct Adam <: Optimizer
    opt_type::String
    t::Int64
    ϵ::Float64
    α::Float64
    β₁::Float64
    β₂::Float64
    m_t::Array{Float64}
    v_t::Array{Float64}
end

"Construct Adam optimizer"
function Adam(;α=0.001, β₁=0.9, β₂=0.999, ϵ=10e-8)
    m_t = [0.0]
    v_t = [0.0]

    Adam("Adam", 0, ϵ, α, β₁, β₂, m_t, v_t)
end

params(opt::Adam) = "ϵ=$(opt.ϵ), α=$(opt.α), β₁=$(opt.β₁), β₂=$(opt.β₂)"

function update(opt::Adam, g_t::Array{Float64})
    # resize biased moment estimates if first iteration
    if opt.t == 0
        opt.m_t = zero(g_t)
        opt.v_t = zero(g_t)
    end

    # update timestep
    opt.t += 1

    # update biased first moment estimate
    opt.m_t = opt.β₁ * opt.m_t + (1. - opt.β₁) * g_t

    # update biased second raw moment estimate
    opt.v_t = opt.β₂ * opt.v_t + (1. - opt.β₂) * ((g_t) .^2)

    # compute bias corrected first moment estimate
    m̂_t = opt.m_t / (1. - opt.β₁^opt.t)

    # compute bias corrected second raw moment estimate
    v̂_t = opt.v_t / (1. - opt.β₂^opt.t)

    # apply update
    ρ = opt.α * m̂_t ./ (sqrt.(v̂_t .+ opt.ϵ))

    return ρ
end
