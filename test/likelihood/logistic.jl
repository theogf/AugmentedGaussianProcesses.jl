@testset "logistic" begin
    N, d = 20, 2
    k = 2.0 * SqExponentialKernel() ∘ ScaleTransform(10.0)
    X, f = generate_f(N, d, k)
    y = f .> 0
    floattypes = [Float64]
    tests_likelihood(
        LogisticLikelihood(),
        BernoulliLikelihood,
        Dict(
            "VGP" => Dict("AVI" => true, "QVI" => true, "MCVI" => false),
            "SVGP" => Dict("AVI" => true, "QVI" => true, "MCVI" => false),
            "OSVGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => false),
            "MCGP" => Dict("Gibbs" => true, "HMC" => false),
        ),
        floattypes,
        "Classification",
        1,
        X,
        f,
        y,
        k,
    )
end
