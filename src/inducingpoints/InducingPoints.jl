module InducingPoints

export AbstractInducingPoints
export KmeansIP
export OptimIP
export Webscale, OIPS, Kmeans, kDPP, StdDPP, SeqDPP, Greedy, UniformSampling, UniGrid
export init, update!

using StatsBase: Weights, sample
using DeterminantalPointProcesses
using LinearAlgebra#: Symmetric, Eigen, eigen, eigvals, I, logdet, diag, norm
using Clustering: kmeans!
using Distances
using DataStructures
using KernelFunctions
using KernelFunctions: ColVecs, RowVecs, vec_of_vecs
using Random: rand, bitrand, AbstractRNG, MersenneTwister
import Base: rand, show

const jitt = 1e-5

abstract type AbstractInducingPoints{S, TZ<:AbstractVector{S}} <: AbstractVector{S} end

const AIP = AbstractInducingPoints

abstract type OfflineInducingPoints{S, TZ<:AbstractVector{S}} <: AIP{S, TZ} end

const OffIP = OfflineInducingPoints

abstract type OnlineInducingPoints{S, TZ<:AbstractVector{S}} <: AIP{S, TZ} end

const OnIP = OnlineInducingPoints

Base.size(Z::AIP) = size(Z.Z)
Base.length(Z::AIP) = length(Z.Z)
Base.getindex(Z::AIP, i::Int) = getindex(Z.Z, i)
Base.vec(Z::AIP) = Z.Z

struct CustomInducingPoints{S,TZ<:AbstractVector{S}} <: OffIP{S,TZ}
     Z::TZ
end

init(Z::OnIP, X::AbstractMatrix, args...; obsdim = 1) = init(Z, vec_of_vecs(X, obsdim = obsdim), args...)

init(Z::OnIP, X::AbstractVector, k::Kernel) = init(Z, X)

update!(Z::OnIP, X::AbstractMatrix, args...; obsdim = 1) = init(Z, vec_of_vecs(X, obsdim = obsdim), args...)

update!(Z::OnIP, X::AbstractVector, args...) = add_point!(Z, X, args...)

add_point!(Z::OnIP, X::AbstractVector, k::Kernel) = add_point!(Z, X)

remove_point!(Z::OnIP, args...) = nothing




include("optimIP.jl")
include("seqdpp.jl")
include("kdpp.jl")
include("stddpp.jl")
include("streamingkmeans.jl")
include("webscale.jl")
include("oips.jl")
include("kmeans.jl")
include("greedy.jl")
include("uniform.jl")
include("unigrid.jl")

setZ!(Z::OptimIP, Zvec::AbstractVector) = OptimIP(setZ!(Z.Z, Zvec), Z.opt)
setZ!(Z::OffIP, Zvec::AbstractVector) = Z.Z .= Zvec
setZ!(Z::AIP, Zvec::AbstractVector) = Z.Z .= Zvec
setZ!(Z::OnIP, Zvec::AbstractVector) = Z.Z = Zvec


end
