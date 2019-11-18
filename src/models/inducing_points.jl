struct InducingPoints{M,O<:Union{Optimizer,Nothing}}
    Z::M
    opt::O
end

Base.size(Z::InducingPoints) = size(Z.Z)
Base.size(Z::InducingPoints,i::Int) = size(Z.Z,i)
Base.convert(::Type{<:AbstractMatrix},Z::InducingPoints) = Z.Z
