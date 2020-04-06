using AugmentedGaussianProcesses
using Test
using Random: seed!
seed!(42)

AGP.setadbackend(:reverse_diff)
# Global flags for the tests
@testset "AugmentedGaussianProcesses.jl tests" begin
    @testset "Functions" begin

    end
    @testset "Hyperparameters" begin

    end
    @testset "Inducing Points" begin
        for f in readdir(joinpath(@__DIR__, "inducingpoints"))
            include(joinpath("inducingpoints",f))
        end
    end
    @testset "Inference" begin
        for f in readdir(joinpath(@__DIR__, "inference"))
            include(joinpath("inference",f))
        end
    end
    @testset "Likelihoods" begin

    end
    @testset "Models" begin

    end
    @testset "Prior" begin
        for f in readdir(joinpath(@__DIR__, "prior"))
            include(joinpath("prior",f))
        end
    end
    @testset "Training" begin

    end

    include("test_likelihoods.jl")
end
