struct InducingPoints{T,M<:AbstractMatrix{T},O<:Union{Optimizer,Nothing}} <: AbstractMatrix{T}
    Z::M
    opt::O
end

function InducingPoints(Z::AbstractMatrix{T},opt::O=nothing) where {T<:Real,O<:Union{Optimizer,Nothing}}
    InducingPoints{T,typeof(Z),O}(Z,opt)
end

Base.size(Z::InducingPoints) = size(Z.Z)
Base.size(Z::InducingPoints,i::Int) = size(Z.Z,i)
Base.getindex(Z::InducingPoints,i::Int) = getindex(Z.Z,i)
Base.getindex(Z::InducingPoints,i::Int,j::Int) = getindex(Z.Z,i,j)
