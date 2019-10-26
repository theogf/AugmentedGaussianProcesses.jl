struct InducingPoints{M,O}
    Z::M
    opt::O
end

Base.size(Z::InducingPoints) = size(Z.Z)
Base.size(Z::InducingPoints,i::Int) = size(Z.Z,i)
