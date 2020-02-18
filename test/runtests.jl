using AugmentedGaussianProcesses
using Test
using Random: seed!
seed!(42)

AGP.setadbackend(:forward_diff)
# Global flags for the tests
@testset "Test for AugmentedGaussianProcesses" begin
include("test_GP.jl")
include("test_VGP.jl")
include("test_SVGP.jl")
# @test include("test_IO.jl")
end
