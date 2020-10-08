"""
    StdDPP(X::AbstractMatrix, kernel::Kernel; obsdim::Int = 1)
    StdDPP(X::AbstractVector, kernel::Kernel)

Standard DPP (Determinantal Point Process) sampling given `kernel`.
The size of the returned `Z` is variable
"""
struct StdDPP{S,TZ<:AbstractVector{S},K<:Kernel} <: OffIP{S,TZ}
    kernel::K
    k::Int
    Z::TZ
end

function StdDPP(X::AbstractVector, kernel::K) where {K<:Kernel}
    Z = stdpp_ip(X, kernel)
    return StdDPP(kernel, length(Z), Z)
end

function stdpp_ip(X, kernel)
    K = Symmetric(kernelmatrix(kernel, X) + jitt * I)
    dpp = DPP(K)
    samp = rand(dpp, 1)[1]
    Z = Vector.(X[samp])
end
