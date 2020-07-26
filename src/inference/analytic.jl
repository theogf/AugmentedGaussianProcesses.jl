## Solve the classical GP Regression ##
mutable struct Analytic{T<:Real,Tx<:AbstractVector,Ty<:AbstractVector} <:
               Inference{T}
    ϵ::T #Convergence criteria
    nIter::Integer #Number of steps performed
    nSamples::Int
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    xview::Tx
    yview::Ty
    function Analytic{T}(ϵ::T) where {T}
        return new{T,Vector{T},Vector{T}}(ϵ)
    end
    function Analytic{T}(
        ϵ::T,
        nSamples::Integer,
        xview::Tx,
        yview::Ty,
    ) where {T,Tx<:AbstractVector,Ty<:AbstractVector}
        return new{T,Tx,Ty}(ϵ, 0, nSamples, true, xview, yview)
    end
end

"""
    Analytic(;ϵ::T=1e-5)

Analytic inference structure for the classical GP regression

**Keyword arguments**
    - `ϵ::T` : convergence criteria, which can be user defined
"""
function Analytic(; ϵ::T = 1e-5) where {T<:Real}
    Analytic{T}(ϵ)
end


function Base.show(io::IO, inference::Analytic{T}) where {T}
    print(io, "Analytic Inference")
end


function init_inference(
    i::Analytic{T},
    nLatent::Integer,
    nFeatures::Integer,
    nSamples::Integer,
    nSamplesUsed::Integer,
    xview::AbstractVector,
    yview::AbstractVector,
) where {T<:Real}
    return Analytic{T}(conv_crit(i), nSamples, xview, yview)
end

function analytic_updates!(m::GP{T}) where {T}
    f = getf(model)
    l = likelihood(model)
    f.Σ = f.K + first(l.σ²) * I
    f.μ .= f.Σ * (get_y(m) / first(l.σ²) - f.K \ f.μ₀(xview(m))
    if !isnothing(l.opt_noise)
        g = 0.5 * (norm(f.μ, 2) - tr(inv(f.Σ)))
        Δlogσ² = Flux.Optimise.apply!(l.opt_noise, l.σ², g .* l.σ²)
        l.σ² .= exp.(log.(l.σ²) .+ Δlogσ²)
    end
end

xview(inf::Analytic) = inf.xview
yview(inf::Analytic) = inf.yview

nMinibatch(inf::Analytic) = inf.nSamples

getρ(inf::Analytic{T}) = one(T)

MBIndices(inf::Analytic) = 1:inf.nSamples

isStochastic(::Analytic) = false
