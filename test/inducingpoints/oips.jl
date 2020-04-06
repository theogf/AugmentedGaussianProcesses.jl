using Test
using AugmentedGaussianProcesses

N = 50
D = 3
nInd = 20
k = transform(SqExponentialKernel(), 10.0)
X = rand(N, D)
y = rand(N)

@testset "OIPS" begin
    ρ_accept = 0.8;
    ρ_remove = 0.9;
    alg = OIPS(ρ_accept, ρ_remove)
    @test_nowarn println(alg)
    AGP.IPModule.init!(alg, X, y, k)

    alg = OIPS(nInd)
    @test_nowarn println(alg)
    AGP.IPModule.init!(alg, X, y, k)
    @test size(alg, 1) <= nInd
end
