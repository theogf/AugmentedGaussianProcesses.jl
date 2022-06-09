@testset "heteroscedastic" begin
    N, d = 500, 1
    k = SqExponentialKernel() ∘ ScaleTransform(10.0)
    X, f = generate_f(N, d, k)
    X, g = generate_f(N, d, k; X=X)
    g .-= 3.0
    λ = 2.0
    σ = λ * AGP.logistic.(g)

    y = f + rand.(Normal.(0.0, sqrt.(inv.(σ))))
    floattypes = [Float64]
    tests_likelihood(
        HeteroscedasticLikelihood(λ),
        AGP.GPLikelihoods.HeteroscedasticGaussianLikelihood,
        Dict(
            "VGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => false),
            "SVGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => false),
            "OSVGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => false),
            "MCGP" => Dict("Gibbs" => true, "HMC" => false),
        ),
        floattypes,
        "Regression",
        2,
        X,
        f,
        y,
        k,
    )
end
