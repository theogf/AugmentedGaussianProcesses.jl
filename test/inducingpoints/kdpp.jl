seed!(42)
N = 20
D = 3
nInd = 10
k = transform(SqExponentialKernel(), 10.0)
X = rand(N, D)
y = rand(N)

@testset "kDPP" begin
    alg = kDPP(nInd, k)
    @test repr(alg) == "kDPP selection of inducing points"
    AGP.IPModule.init!(alg, X, y, k)
    @test size(alg) == (nInd, D)
end
