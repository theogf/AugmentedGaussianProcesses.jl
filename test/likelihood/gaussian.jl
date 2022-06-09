@testset "gaussian" begin
    seed!(42)
    @testset "Likelihood" begin
        σ² = 1e-3
        l = GaussianLikelihood(σ²)
        f = 0.5
        y = 0.2
        @test AGP.noise(l) == σ²
        @test l(y, f) == pdf(Normal(f, sqrt(σ²)), y)
        @test loglikelihood(l, y, f) == logpdf(Normal(f, sqrt(σ²)), y)
        @test repr(l) == "Gaussian likelihood (σ² = $(AGP.noise(l)))"
        @test AGP.AugmentedKL(l, [], (;)) == 0.0
        @test AGP.n_latent(l) == 1
    end
    N, d = 20, 2
    k = SqExponentialKernel() ∘ ScaleTransform(10.0)
    σ = 0.1
    X, f = generate_f(N, d, k)
    y = f + σ * randn(N)
    floattypes = [Float64]
    @testset "GP" begin
        for floattype in floattypes
            model = GP(X, y, k; opt_noise=true, verbose=0)
            @test eltype(model) == floattype
            @test AGP.likelihood(model) isa GaussianLikelihood
            @test AGP.inference(model) isa Analytic
            @test AGP.getf(model) isa AGP.LatentGP
            @test AGP.n_latent(model) == 1
            L = AGP.objective(model, X, y)
            train!(model, 10)
            @test L < AGP.objective(model, X, y)
            @test testconv(model, "Regression", X, f, y)
            @test all(proba_y(model, X)[2] .> 0)
        end
    end
    @testset "VGP" begin
        @test_throws ErrorException VGP(X, y, k, GaussianLikelihood(), AnalyticVI())
        @test_throws ErrorException VGP(X, y, k, GaussianLikelihood(), QuadratureVI())
        @test_throws ErrorException VGP(X, y, k, GaussianLikelihood(), MCIntegrationVI())
    end
    @testset "SVGP" begin
        Z = inducingpoints(KmeansAlg(M), X)
        for floattype in floattypes
            test_inference_SVGP(
                X,
                y,
                f,
                k,
                GaussianLikelihood(),
                GaussianLikelihood,
                floattype,
                Z,
                1,
                "Regression",
                AnalyticVI(),
                AnalyticSVI(10);
                valid=true,
            )
            @test_throws ErrorException SVGP(k, GaussianLikelihood(), QuadratureVI(), Z)
            @test_throws ErrorException SVGP(k, GaussianLikelihood(), MCIntegrationVI(), Z)
        end
    end
    @testset "MCGP" begin
        @test_throws ErrorException MCGP(X, y, k, GaussianLikelihood(), GibbsSampling())
        @test_throws ErrorException MCGP(X, y, k, GaussianLikelihood(), HMCSampling())
    end
end
