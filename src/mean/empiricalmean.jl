struct EmpiricalMean{T<:Real,V<:AbstractVector{<:Real},O} <: PriorMean{T}
    C::V
    opt::O
end

"""
    EmpiricalMean(c::AbstractVector{<:Real}=1.0;opt=ADAM(0.01))

## Arguments
- `c::AbstractVector` : Empirical mean vector

Construct a empirical mean with values `c`
Optionally give an optimiser `opt` (`ADAM(0.01)` by default)
"""
function EmpiricalMean(c::V; opt=ADAM(0.01)) where {V<:AbstractVector{<:Real}}
    return EmpiricalMean{eltype(c),V,typeof(opt)}(copy(c), opt)
end

function Base.show(io::IO, ::MIME"text/plain", μ₀::EmpiricalMean)
    return print(io, "Empirical Mean Prior (length(c) = ", length(μ₀.C), ")")
end

function (μ₀::EmpiricalMean{T})(x::AbstractMatrix) where {T<:Real}
    size(x, 1) == length(μ₀.C) || error("Wrong dimension between x and μ₀")
    return μ₀.C
end

function update!(μ₀::EmpiricalMean{T}, grad) where {T<:Real}
    return μ₀.C .+= Optimise.apply!(μ₀.opt, μ₀.C, grad.C)
end
