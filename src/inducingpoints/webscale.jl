"""
    Webscale(k::Int)

Online k-means algorithm based on [1].

[1] Sculley, D. Web-scale k-means clustering. in Proceedings of the 19th international conference on World wide web - WWW ’10 1177 (ACM Press, 2010). doi:10.1145/1772690.1772862.
"""
mutable struct Webscale{S,TZ<:AbstractVector{S}} <: OnIP{S,TZ}
    k::Int
    v::Vector{Int}
    Z::TZ
end

function Webscale(k::Int)
    return Webscale(k, [], [])
end

function Webscale(Z::Webscale, X::AbstractVector)
    size(X, 1) >= Z.k || error("Input data not big enough given desired number of inducing points : $(Z.k)")
    v = zeros(Int, Z.k)
    Z = Vector.(X[sample(1:size(X, 1), alg.k, replace = false)])
    return Webscale(Z.k, v, Z)
end

function init(Z::Webscale, X::AbstractVector)
    return Webscale(Z, X)
end

function add_point!(alg::Webscale, X::AbstractVector)
    b = size(X, 1)
    d = zeros(Int64, b)
    for i = 1:b
        d[i] = find_nearest_center(X[i], Z)[1]
    end
    for i = 1:b
        Z.v[d[i]] += 1
        η = 1 / Z.v[d[i]]
        Z[d[i]] = (1 - η) * Z[d[i]] + η * X[i]
    end
end
