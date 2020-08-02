N = 50

x = rand(N)
X = collect(reshape(x, :, 1))
y = rand(N)

@testset "Variational Student-T Processes" begin
    ν = 3.0
    k = SqExponentialKernel()
    l = GaussianLikelihood()
    vi = AnalyticVI()
    m = VStP(x, y, k, l, vi, ν)
    @test_throws ErrorException VStP(x, y, k, l, vi, 0.5)
    @test_throws ErrorException VStP(x, y, k, l, QuadratureVI(), 0.5)
    @test_nowarn println(m)
    AGP.computeMatrices!(m)
    @test_nowarn AGP.local_prior_updates!(m, collect(eachrow(X)))
    @test AGP.objective(m) == ELBO(m)
    # @test AGP.Zview(m) == [X]
    # @test AGP.get_X(m) == X
    # @test AGP.Zview(m, 1) == X

    train!(m, 20)

end
