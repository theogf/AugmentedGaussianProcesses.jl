using AugmentedGaussianProcesses
using Test
using Random: seed!
seed!(42)

include("testingtools.jl")

AGP.setadbackend(:reverse_diff)
# Global flags for the tests
@testset "AugmentedGaussianProcesses.jl tests" begin
    @info "Function tests"
    @testset "Functions" begin
        for f in readdir(joinpath(@__DIR__, "functions"))
            include(joinpath("functions",f))
        end
    end

    @info "Hyperparameter tests"
    @testset "Hyperparameter`s" begin

    end

    @info "Inference tests"
    @testset "Inference" begin
        for f in readdir(joinpath(@__DIR__, "inference"))
            include(joinpath("inference", f))
        end
    end

    @info "Likelihood tests"
    @testset "Likelihoods" begin
        for f in readdir(joinpath(@__DIR__, "likelihood"))
            include(joinpath("likelihood",f))
        end
    end

    @info "Model tests"
    @testset "Models" begin
        for f in readdir(joinpath(@__DIR__, "models"))
            include(joinpath("models",f))
        end
    end

    @info "Prior tests"
    @testset "Prior" begin
        for f in readdir(joinpath(@__DIR__, "prior"))
            include(joinpath("prior",f))
        end
    end
    include("training.jl")
    include("onlinetraining.jl")
    include("predictions.jl")
end
