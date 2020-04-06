using Test
using AugmentedGaussianProcesses
using Distributions


L = 3
D = 10
N = 20
nSamples = 10
b = 5
x = rand(N, D)
y = rand(N)
@testset "Inference" begin
    i = AnalyticVI()
    @test length(i) == 1
end
