using Test
using AugmentedGaussianProcesses
using LinearAlgebra
using Statistics
const AGP = AugmentedGaussianProcesses
include("testingtools.jl")

nData = 100; nDim = 2
m = 50; b= 10
k = AGP.RBFKernel()

X = rand(nData,nDim)
y = norm.(eachrow(X))
floattypes = [Float64]
@testset "GP" begin
    for floattype in floattypes
        @test typeof(GP(X,y,k)) <: GP{eval(Meta.parse("GaussianLikelihood{"*string(floattype)*"}")),eval(Meta.parse("AnalyticInference{"*string(floattype)*"}")),floattype,Vector{floattype}}
        model = GP(X,y,k,Autotuning=true,verbose=3)
        @test train!(model,iterations=50)
        @test mean(abs2.(predict_y(model,X)-y)) < 0.2
    end
end
