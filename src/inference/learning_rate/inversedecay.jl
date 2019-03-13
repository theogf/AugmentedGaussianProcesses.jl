""" Learning rate for scheme (τ+t)^-κ """
mutable struct InverseDecay <: GradDescent.Optimizer
    opt_type::String
    t::Int64
    κ::Float64
    τ::Int64
    ρ::Float64
end

""" Construct Learning Rate Scheme satisfying Robbins-Monro conditions"""
function InverseDecay(;τ::Int=100,κ::Real=0.51)
    InverseDecay("Inverse Decay",0,κ,τ,-1.0)
end

params(opt::InverseDecay) = "τ=$(opt.τ), κ=$(opt.κ)"

function GradDescent.update(opt::InverseDecay, g_t::AbstractArray{T,N}) where {T<:Real,N}
    opt.t += 1
    opt.ρ = (opt.t+opt.τ)^(-opt.κ)
    return opt.ρ * g_t
end
