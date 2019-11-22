struct InducingPoints{T,M,O<:Union{Optimizer,Nothing}} <: AbstractMatrix{T}
    Z::M
    opt::O
end

Base.size(Z::InducingPoints) = size(Z.Z)
Base.size(Z::InducingPoints,i::Int) = size(Z.Z,i)
Base.convert(::Type{<:AbstractMatrix},Z::InducingPoints) = Z.Z
Base.getindex(Z::InducingPoints,i::Int) = getindex(Z.Z[i])
Base.getindex(Z::InducingPoints,i::Int,j::Int) = getindex(Z.Z[i,j])
