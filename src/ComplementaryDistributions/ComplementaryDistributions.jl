module ComplementaryDistributions

using Distributions
using Random
using SpecialFunctions
using StatsFuns: twoπ, halfπ, inv2π, fourinvπ

export GeneralizedInverseGaussian, PolyaGamma, LaplaceTransformDistribution
include("generalizedinversegaussian.jl")
include("polyagamma.jl")
include("lap_transf_dist.jl")

end
