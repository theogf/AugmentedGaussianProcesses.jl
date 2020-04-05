using AugmentedGaussianProcesses
using Test
using Random: seed!
seed!(42)

AGP.setadbackend(:reverse_diff)
# Global flags for the tests
@testset "AugmentedGaussianProcesses.jl tests" begin
    @testset "Prior" begin
        for f in readdir("prior")
            include(joinpath("prior",f))
        end
    end
    include("test_prior.jl")
    include("test_likelihoods.jl")
    include("test_inference.jl")
# include("test_analyticVI.jl")
# include("test_SVGP.jl")
# include("test_OnlineSVGP.jl")
# @test include("test_IO.jl")
end
