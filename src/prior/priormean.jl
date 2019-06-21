abstract type PriorMean{T} end

import Base: +, -, *, convert

include("constantmean.jl")
include("zeromean.jl")
include("empiricalmean.jl")

function Base.convert(::Type{PriorMean},x::T) where {T<:Real}
    ConstantMean(x)
end

function Base.convert(::Type{PriorMean},x::AbstractVector{T}) where {T<:Real}
    EmpiricalMean(x)
end
