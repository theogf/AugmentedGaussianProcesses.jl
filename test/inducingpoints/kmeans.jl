using Test
using AugmentedGaussianProcesses

N = 50
D = 3
nInd = 10
k = transform(SqExponentialKernel(), 10.0)
X = rand(N, D)
y = rand(N)

@testset "k-Means" begin
    alg = Kmeans(nInd)
    @test_nowarn println(alg)
    AGP.IPModule.init!(alg, X, y, k)
    @test size(alg) == (nInd, D)
end
