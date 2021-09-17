abstract type PriorMean{T} end

import Base: convert

include("constantmean.jl")
include("zeromean.jl")
include("empiricalmean.jl")
include("affinemean.jl")

function Base.convert(::Type{PriorMean}, x::T) where {T<:Real}
    return ConstantMean(x)
end

function Base.convert(::Type{PriorMean}, x::AbstractVector{T}) where {T<:Real}
    return EmpiricalMean(x)
end

function init_priormean_state(hyperopt_state, ::PriorMean)
    return hyperopt_state
end
