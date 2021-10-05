@testset "bayesiansvm" begin
    seed!(42)
    N, d = 20, 2
    k = SqExponentialKernel() ∘ ScaleTransform(10.0)
    X, f = generate_f(N, d, k)
    y = f .> 0
    floattypes = [Float64]
    tests_likelihood(
        BayesianSVM(),
        BernoulliLikelihood,
        Dict(
            "VGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => false),
            "SVGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => false),
            "OSVGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => false),
            "MCGP" => Dict("Gibbs" => false, "HMC" => false),
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
