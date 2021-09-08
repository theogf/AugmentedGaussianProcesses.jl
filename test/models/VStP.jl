@testset "Variational Student-T Processes" begin
    N = 50

    x = rand(N)
    X = collect(reshape(x, :, 1))
    y = rand(N)
    ν = 3.0
    k = SqExponentialKernel()
    l = GaussianLikelihood()
    vi = AnalyticVI()
    m = VStP(x, y, k, l, vi, ν)
    @test_throws ErrorException VStP(x, y, k, l, vi, 0.5)
    @test_throws ErrorException VStP(x, y, k, l, QuadratureVI(), 0.5)
    @test_nowarn println(m)
    AGP.compute_kernel_matrices!(m)
    @test_nowarn AGP.local_prior_updates!(m, collect(eachrow(X)))
    @test AGP.objective(m) == ELBO(m)

    train!(m, 20)
end
