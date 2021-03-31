## Solve the classical GP Regression ##
mutable struct Analytic{T<:Real,Tx<:AbstractVector,Ty<:AbstractVector} <:
               AbstractInference{T}
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
        ϵ::T, nSamples::Integer, xview::Tx, yview::Ty
    ) where {T,Tx<:AbstractVector,Ty<:AbstractVector}
        return new{T,Tx,Ty}(ϵ, 0, nSamples, true, xview, yview)
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

function analytic_updates!(m::GP{T}) where {T}
    f = getf(m)
    l = likelihood(m)
    f.post.Σ = pr_cov(f) + first(l.σ²) * I
    f.post.α .= cov(f) \ (yview(m) - pr_mean(f, xview(m)))
    if !isnothing(l.opt_noise)
        g = 0.5 * (norm(mean(f), 2) - tr(inv(cov(f))))
        Δlogσ² = Optimise.apply!(l.opt_noise, l.σ², g .* l.σ²)
        l.σ² .= exp.(log.(l.σ²) .+ Δlogσ²)
    end
end

xview(i::Analytic) = i.xview
yview(i::Analytic) = i.yview

nMinibatch(i::Analytic) = i.nSamples

getρ(::Analytic{T}) where {T} = one(T)

MBIndices(i::Analytic) = 1:nSamples(i)

isStochastic(::Analytic) = false
