mutable struct Momentum <: Optimizer
    opt_type::String
    t::Int64
    η::Float64
    γ::Float64
    v_t::Array{Float64}
end

"Construct Momentum optimizer"
function Momentum(; η::Float64=0.01, γ::Float64=0.9)
    η <= 0.0 && error("η must be greater than 0")
    γ <= 0.0 && error("γ must be greater than 0")

    Momentum("Momentum", 0, η, γ, [0.0])
end

params(opt::Momentum) = "η=$(opt.η), γ=$(opt.γ)"

function update(opt::Momentum, g_t::Array{Float64})
    # resize squares of gradients
    if opt.t == 0
        opt.v_t = zero(g_t)
    end

    # update timestep
    opt.t += 1

    opt.v_t = γ * opt.v_t + opt.η * g_t

    return opt.v_t
end
