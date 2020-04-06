"""
    Module for methods to find inducing points
"""
module IPModule

export Webscale, OIPS, Kmeans, kDPP, StdDPP, SeqDPP, Greedy, UniformSampling, InducingPoints, FixedInducingPoints
export init!, add_point!, remove_point!

using StatsBase: Weights, sample
using DeterminantalPointProcesses
using LinearAlgebra#: Symmetric, Eigen, eigen, eigvals, I, logdet, diag, norm
using Clustering: kmeans!
using Distances
using DataStructures
using KernelFunctions
using Random: rand, bitrand, AbstractRNG, MersenneTwister
using Flux.Optimise
import Base: rand

const jitt = 1e-5

include("inducing_points.jl")
include("seqdpp.jl")
include("kdpp.jl")
include("stddpp.jl")
include("streamingkmeans.jl")
include("webscale.jl")
include("oips.jl")
include("kmeans.jl")
include("greedy.jl")
include("uniform.jl")



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
