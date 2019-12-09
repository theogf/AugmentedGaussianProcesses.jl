using Test
using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses
using LinearAlgebra
using Statistics
using Distributions
include("testingtools.jl")

nData = 100; nDim = 2
m = 50; b= 10
k = SqExponentialKernel(10.0)

X = rand(nData,nDim)
y = rand(MvNormal(zeros(nData),kernelmatrix(k,X,obsdim=1)+1e-3I))

floattypes = [Float64]
@testset "GP Testing" begin
    for floattype in floattypes
        @test typeof(GP(X,y,k)) <: GP{floattype,eval(Meta.parse("GaussianLikelihood{"*string(floattype)*"}")),eval(Meta.parse("Analytic{"*string(floattype)*"}")),AGP._GP{floattype},1}
        model = GP(X,y,k,verbose=2)
        @test train!(model,50)
        @test mean(abs2.(predict_y(model,X)-y)) < 0.2
    end
end
