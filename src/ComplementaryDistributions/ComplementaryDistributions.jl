module ComplementaryDistributions

using Distributions
using Random
using SpecialFunctions

export GeneralizedInverseGaussian, PolyaGamma, LaplaceTransformDistribution
include("generalizedinversegaussian.jl")
include("polyagamma.jl")
include("lap_transf_dist.jl")

end