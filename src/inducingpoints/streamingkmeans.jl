##From paper "An Algorithm for Online K-Means Clustering" ##
"""
    StreamOnline(k_target::Int)

Online clustering algorithm [1] to select inducing points in a streaming setting.
Reference :
[1] Liberty, E., Sriharsha, R. & Sviridenko, M. An Algorithm for Online K-Means Clustering. arXiv:1412.5721 [cs] (2015).
"""
mutable struct StreamOnline{S,TZ<:AbstractVector{S}} <: OnIP{S,TZ}
    k_target::Int
    k_efficient::Int
    k::Int
    f::Float64
    q::Int
    Z::TZ
end

StreamOnline(k_target::Int) = StreamOnline(k_target, 0, 0, 0.0, 0, [])

function StreamOnline(Z::StreamOnline, X::AbstractVector, kernel = nothing)
    size(X, 1) > 10 ||
        error("The first batch of data should be bigger than 10 samples")
    k_efficient = max(1, ceil(Int64, (Z.k_target - 15) / 5))
    if k_efficient + 10 > size(X, 1)
        k_efficient = 0
    end
    samp = sample(1:size(X, 1), alg.k_efficient + 10, replace = false)
    Z = Vector.(X[samp])
    k = k_efficient + 10
    w = zeros(k)
    for i = 1:k
        w[i] = 0.5 * find_nearest_center(Z[i], Z[1:alg.k.!=i], kernel)[2]
    end
    f = sum(sort(w)[1:10]) #Take the 10 smallest values
    q = 0
    StreamOnline(Z.k_target, k_efficient, k, f, q, Z)
end

function init(Z::StreamOnline, X::AbstractVector)
    Z = StreamOnline(Z, X)
end

function add_point!(alg::StreamOnline, X::AbstractVector, kernel = nothing)
    b = size(X, 1)
    for i = 1:b
        val = find_nearest_center(X[i], Z, kernel)[2]
        if val > (Z.f * rand())
            # new_centers = vcat(new_centers,X[i,:]')
            Z.Z = vcat(Z.Z, Vector(X[i]))
            Z.q += 1
            Z.k += 1
        end
        if Z.q >= Z.k_efficient
            Z.q = 0
            Z.f *= 10
        end
    end
end

"Find the closest center to X among C, return the index and the distance"
function find_nearest_center(X::AbstractVector, C::AbstractVector, kernel = nothing)
    nC = size(C, 1)
    best = Int64(1)
    best_val = Inf
    for i = 1:nC
        val = distance(X, C[i], kernel)
        if val < best_val
            best_val = val
            best = i
        end
    end
    return best, best_val
end

"Compute the distance (kernel if included) between a point and a find_nearest_center"
function distance(X, C, k = nothing)
    if isnothing(k)
        return norm(X - C, 2)^2
    else
        k(X, C)
    end
end
