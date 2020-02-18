"""
    Module for methods to find inducing points
"""
module IPModule

using Distributions, StatsBase
using LinearAlgebra, Clustering, Distances, DataStructures
using KernelFunctions
using DeterminantalPointProcesses
using Flux: Optimise
export Webscale, OIPS, Kmeans, kDPP, StdDPP, SeqDPP, Greedy, UniformSampling, InducingPoints, FixedInducingPoints
export init!, add_point!, remove_point!

include("inducing_points.jl")
include("dpp_base.jl")
include("SeqDPP.jl")
include("KDPP.jl")
include("StdDPP.jl")
include("StreamingKMeans.jl")
include("Webscale.jl")
include("OIPS.jl")
include("KMeans.jl")
include("greedy.jl")
include("Uniform.jl")

# #Compute the minimum distance
# function mindistance(x::AbstractArray{T,N1},C::AbstractArray{T,N2},nC::Integer) where {T,N1,N2}#Point to look for, collection of centers, number of centers computed
#   mindist = Inf
#   for i in 1:nC
#     mindist = min.(norm(x.-C[i])^2,mindist)
#   end
#   return mindist
# end
#
#
#
# "Return the total cost of the current dataset given a set of centers"
# function total_cost(X,C,kernel)
#     n = size(X,1)
#     tot = 0
#     for i in 1:n
#         tot += find_nearest_center(X[i,:],C,kernel)[2]
#     end
#     return tot
# end
#
# "Return the total cost of the current algorithm"
# function total_cost(X::Array{Float64,2},alg::ZAlg,kernel=0)
#     n = size(X,1)
#     tot = 0
#     for i in 1:n
#         tot += find_nearest_center(X[i,:],alg.centers,kernel)[2]
#     end
#     return tot
# end

end #module
