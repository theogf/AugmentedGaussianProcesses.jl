"""
    UniformSampling(X::AbstractVector, m::Int; weights)
    UniformSampling(X::AbstractMatrix, m::int; weights, obsdim = 1)

Uniform sampling of a subset of the data.
"""
struct UniformSampling{S,TZ<:AbstractVector{S}} <: OffIP{S,TZ}
    k::Int
    Z::TZ
end

function UniformSampling(X::AbstractVector, m::Int; weights = nothing)
    UniformSampling(m, uniformsamplig_ip(X, m, weights))
end

function uniformsampling(X::AbstractVector, m::Int, weights)
    N = size(X, 1)
    N >= m || "Input data not big enough given $k"
    samp = if isnothing(weights)
        sample(1:N, m, replace = false)
    else
        sample(1:N, m, replace = false, weights = weights)
    end
    Z = Vector.(X[samp])
end
