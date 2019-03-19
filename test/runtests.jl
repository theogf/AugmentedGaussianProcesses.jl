using AugmentedGaussianProcesses
using Test
using Random: seed!
seed!(42)

# Global flags for the tests
include("test_kernel.jl")
include("test_GP.jl")
include("test_VGP.jl")
include("test_SVGP.jl")
# @test include("test_IO.jl")
