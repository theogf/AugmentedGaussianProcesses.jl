mutable struct Adamax <: Optimizer
    opt_type::String
    t::Int64
    ϵ::Float64
    α::Float64
    β₁::Float64
    β₂::Float64
    m_t::Array{Float64}
    u_t::Array{Float64}
end

"Construct Adamax optimizer"
function Adamax(;α=0.002, β₁=0.9, β₂=0.999, ϵ=10e-8)
    m_t = zeros(1)'
    u_t = zeros(1)'

    Adamax("Adamax", 0, ϵ, α, β₁, β₂, m_t, u_t)
end

params(opt::Adamax) = "ϵ=$(opt.ϵ), α=$(opt.α), β₁=$(opt.β₁), β₂=$(opt.β₂)"

function update(opt::Adamax, g_t::Array{Float64})
    # resize biased moment estimates if first iteration
    if opt.t == 0
        opt.m_t = zeros(g_t)
        opt.u_t = zeros(g_t)
    end

    # update timestep
    opt.t += 1

    # update biased first moment estimate
    opt.m_t = opt.β₁ * opt.m_t + (1. - opt.β₁) * g_t

    # update the exponentially weighted infinity norm
    opt.u_t = max.(opt.β₂ * opt.u_t, abs.(g_t))

    # update parameters
    ρ = (opt.α / (1- opt.β₁^opt.t)) * opt.m_t ./ opt.u_t

    return ρ
end
