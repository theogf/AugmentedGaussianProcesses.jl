mutable struct Adagrad <: Optimizer
    opt_type::String
    t::Int64
    ϵ::Float64
    η::Float64
    G_t::Array{Float64}
end

"Construct Adagrad optimizer"
function Adagrad(; η::Float64=0.01, ϵ::Float64=1e-8)
    η <= 0.0 && error("η must be greater than 0")
    ϵ <= 0.0 && error("ϵ must be greater than 0")

    Adagrad("Adagrad", 0, ϵ, η, [0.0])
end

params(opt::Adagrad) = "ϵ=$(opt.ϵ), η=$(opt.η)"

function update(opt::Adagrad, g_t::Array{Float64})
    # resize squares of gradients
    if opt.t == 0
        opt.G_t = zero(g_t)
    end

    # update timestep
    opt.t += 1

    # accumulate squares of gradients
    opt.G_t += (g_t .^ 2)

    δ = opt.η ./ (sqrt.(opt.G_t + opt.ϵ)) .* g_t

    return δ
end
