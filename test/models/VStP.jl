@testset "Variational Student-T Processes" begin
    N = 50

    x = rand(N)
    X = collect(reshape(x, :, 1))
    y = rand(N)
    ν = 3.0
    k = SqExponentialKernel()
    l = LaplaceLikelihood()
    vi = AnalyticVI()
    m = VStP(x, y, k, l, vi, ν)
    @test_throws ErrorException VStP(x, y, k, l, vi, 0.5)
    @test_throws ErrorException VStP(x, y, k, l, QuadratureVI(), 0.5)
    @test repr(m) ==
        "Variational Student-T Process with a Laplace likelihood (β=1.0) infered by Analytic Variational Inference "
    train!(m, 20)
end
