using AugmentedGaussianProcesses
using Test
using Random: seed!
seed!(42)

# Global flags for the tests
@testset "Augmented Gaussian Process Testing"
include("test_GP.jl")
include("test_VGP.jl")
include("test_SVGP.jl")
end
# @test include("test_IO.jl")
