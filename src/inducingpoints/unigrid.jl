"""
    UniGrid(K)

where `K` is the number of points on each dimension
Adaptive uniform grid.
"""
mutable struct UniGrid{T,M<:AbstractMatrix{T},O} <: InducingPoints{T,M,O}
    K::Int
    opt::O
    D::Int
    bounds::Vector{Vector{Float64}}
    k::Int
    Z::M
    function UniGrid(
        K::Int
    )
        K > 0 || error("K should be positive")
        return new{Float64,Matrix{Float64},Nothing}(
            K, nothing
        )
    end
end

Base.show(io::IO, alg::UniGrid) = print(
    io,
    "Uniform grid with side length $(alg.K).",
)

function init!(alg::UniGrid, X, y, kernel)
    alg.bounds = vec(collect.(extrema(X, dims = 1)))
    alg.opt = nothing
    alg.D = length(alg.bounds)
    ranges = map(alg.bounds) do lims
        LinRange(lims..., alg.K)
    end
    alg.k = alg.K ^ alg.D
    alg.Z = zeros(Float64, alg.k, alg.D)
    for (i, vals) in enumerate(Iterators.product(ranges...))
        alg.Z[i, :] .= vals
    end
end

function add_point!(alg::UniGrid, X, y, kernel)
    new_bounds = extrema(X, dims = 1)
    ranges = map(alg.bounds, new_bounds) do old_b, new_b
        old_b .= min(old_b[1], new_b[1]), max(old_b[1], new_b[1])
        LinRange(old_b..., alg.K)
    end
    for (i, vals) in enumerate(Iterators.product(ranges...))
        alg.Z[i, :] .= vals
    end
end
