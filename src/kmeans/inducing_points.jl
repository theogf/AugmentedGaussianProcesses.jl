abstract type InducingPoints{T,M<:AbstractMatrix{T},O} <: AbstractMatrix{T} end


struct FixedInducingPoints{T,M<:AbstractMatrix{T},O} <: InducingPoints{T,M,O}
    Z::M
    opt::O
end

function FixedInducingPoints(Z::AbstractMatrix{T},opt=nothing) where {T<:Real}
    FixedInducingPoints{T,typeof(Z),typeof(opt)}(Z,opt)
end


Base.size(Z::InducingPoints) = size(Z.Z)
Base.size(Z::InducingPoints,i::Int) = size(Z.Z,i)
Base.getindex(Z::InducingPoints,i::Int) = getindex(Z.Z,i)
Base.getindex(Z::InducingPoints,i::Int,j::Int) = getindex(Z.Z,i,j)
