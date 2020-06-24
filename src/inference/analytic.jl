## Solve the classical GP Regression ##
mutable struct Analytic{T<:Real} <: Inference{T}
    ϵ::T #Convergence criteria
    nIter::Integer #Number of steps performed
    Stochastic::Bool #Use of mini-batches
    nSamples::Int64
    nMinibatch::Int64 #Size of mini-batches
    MBIndices::Vector{Int64} #Indices of the minibatch
    ρ::T #Stochastic Coefficient
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    xview::AbstractArray
    yview::AbstractVector
    function Analytic{T}(
        ϵ::T,
        nIter::Integer,
        Stochastic::Bool,
        nSamples::Integer,
        MBIndices::AbstractVector,
        ρ::T,
    ) where {T}
        return new{T}(ϵ, nIter, Stochastic, nSamples, nSamples, MBIndices, ρ, true)
    end
end

"""
    Analytic(;ϵ::T=1e-5)

Analytic inference structure for the classical GP regression

**Keyword arguments**
    - `ϵ::T` : convergence criteria, which can be user defined
"""
function Analytic(; ϵ::T = 1e-5) where {T<:Real}
    Analytic{T}(ϵ, 0, false, 1, collect(1:1), T(1.0))
end


function Base.show(io::IO,inference::Analytic{T}) where T
    print(io,"Analytic Inference")
end


function init_inference(
    inference::Analytic{T},
    nLatent::Integer,
    nFeatures::Integer,
    nSamples::Integer,
    nSamplesUsed::Integer,
) where {T<:Real}
    inference.nSamples = nSamples
    inference.nMinibatch = nSamples
    inference.MBIndices = 1:nSamples
    inference.ρ = one(T)
    return inference
end

function analytic_updates!(model::GP{T}) where {T}
    f = first(model.f); l = model.likelihood
    f.Σ = f.K + first(l.σ²) * I
    f.μ = f.Σ \ (model.y - f.μ₀(model.X))
    if !isnothing(l.opt_noise)
        g = 0.5 * (norm(f.μ, 2) - tr(inv(f.Σ).mat))
        Δlogσ² = Flux.Optimise.apply!(l.opt_noise, l.σ², g .* l.σ²)
        l.σ² .= exp.(log.(l.σ²) .+ Δlogσ²)
        # mean(abs2, model.y .- f.μ)
    end
end

xview(inf::Analytic) = inf.xview
yview(inf::Analytic) = inf.yview

nMinibatch(inf::Analytic) = inf.nMinibatch

getρ(inf::Analytic) = inf.ρ

MBIndices(inf::Analytic) = inf.MBIndices
