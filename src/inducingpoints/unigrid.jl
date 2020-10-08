"""
    UniGrid(K::Int)

where `K` is the number of points on each dimension
Adaptive uniform grid based on [1]

[1] Moreno-Muñoz, P., Artés-Rodríguez, A. & Álvarez, M. A. Continual Multi-task Gaussian Processes. (2019).
"""
mutable struct UniGrid{S,TZ<:AbstractVector{S}} <: AIP{S,TZ}
    K::Int
    D::Int
    bounds::Vector{Vector{Float64}}
    k::Int
    Z::TZ
end

Base.show(io::IO, Z::UniGrid) = print(
    io,
    "Uniform grid with side length $(Z.K).",
)

function UniGrid(Z::UniGrid, X::AbstractVector)
    d = size(first(X), 1)
    bounds = [extrema(x->getindex(x, i), X) for i in 1:d]
    Z = map(Z.bounds) do lims
        LinRange(lims..., Z.K)
    end
    k = Z.K ^ d
    return UniGrid(Z.K, d, bounds, k, Z)
end

function init(Z::UniGrid, X::AbstractVector)
    return UniGrid(Z, X)
end

function add_point!(Z::UniGrid, X::AbstractVector)
    d = size(first(X), 1)
    new_bounds = [extrema(x->getindex(x, i), X) for i in 1:d]
    Z.Z = map(Z.bounds, new_bounds) do old_b, new_b
        old_b .= min(old_b[1], new_b[1]), max(old_b[2], new_b[2])
        LinRange(old_b..., Z.K) # readapt bounds
    end
end
