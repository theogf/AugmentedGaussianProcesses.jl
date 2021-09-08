@testset "poisson" begin
    N, d = 20, 2
    λ = 5.0
    k = SqExponentialKernel() ∘ ScaleTransform(10.0)
    X, f = generate_f(N, d, k)
    y = rand.(Poisson.(λ * AGP.logistic.(f)))
    floattypes = [Float64]
    tests_likelihood(
        PoissonLikelihood(λ),
        PoissonLikelihood,
        Dict(
            "VGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => false),
            "SVGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => false),
            "OSVGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => false),
            "MCGP" => Dict("Gibbs" => true, "HMC" => false),
        ),
        floattypes,
        "Poisson",
        1,
        X,
        f,
        y,
        k,
    )
end
