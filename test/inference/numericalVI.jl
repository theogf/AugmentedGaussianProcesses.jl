using Test
using AugmentedGaussianProcesses

L = 3
D = 10
N = 20
nSamples = 10
b = 5
x = rand(N, D)
y = rand(N)

@testset "NumericalVI" begin
    @test NumericalVI(:quad) isa QuadratureVI
    @test NumericalVI(:mc) isa MCIntegrationVI
    @test_throws ErrorException NumericalVI(:blah)
    @test NumericalSVI(b,:quad) isa QuadratureVI
    @test NumericalSVI(b,:mc) isa MCIntegrationVI
    @test_throws ErrorException NumericalSVI(b,:blah)
    @test_nowarn show(NumericalVI(:quad))
end
