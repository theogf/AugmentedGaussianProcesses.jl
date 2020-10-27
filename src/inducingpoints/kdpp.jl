"""
    kDPP(X::AbstractVector, m::Int, kernel::Kernel)
    kDPP(X::AbstractMatrix, m::Int, kernel::Kernel; obsdim::Int = 1)

k-DPP (Determinantal Point Process) will return a subset of `X` of size `m`,
according to DPP probability
"""
struct kDPP{S,TZ<:AbstractVector{S},K<:Kernel} <: OffIP{S,TZ}
    k::Int
    kernel::K
    Z::TZ
end

function kDPP(X::AbstractMatrix, m::Int, kernel::Kernel; obsdim::Int = 1)
    kDPP(KernelFunctions.vec_of_vecs(X, obsdim=obsdim), m, kernel)
end

function kDPP(X::AbstractVector, m::Int, kernel::Kernel)
    Z = kddp_ip(X, m, kernel)
    return kDPP(m, kernel, Z)
end

Base.show(io::IO, alg::kDPP) = print(io, "k-DPP selection of inducing points")

function kdpp_ip(X::AbstractVector, m::Int, kernel::Kernel)
    N = size(X, 1)
    Z = Vector{eltype(X)}()
    i = rand(1:N)
    push!(Z, Vector(X[i]))
    IP_set = Set(i)
    k = 1
    kᵢᵢ = kerneldiagmatrix(kernel, X) .+ jitt
    while k < m
        X_set = setdiff(1:N, IP_set)
        kᵢZ = kernelmatrix(kernel, X[collect(X_set)], Z)
        KZ = kernelmatrix(kernel, Z) + jitt * I
        Vᵢ = kᵢᵢ[collect(X_set)] - diag(kᵢZ * inv(KZ) * kᵢZ')
        pᵢ = Vᵢ / sum(Vᵢ)
        j = sample(collect(X_set), Weights(pᵢ))
        push!(Z, Vector(X[j])); push!(IP_set, j)
        k += 1
    end
end
