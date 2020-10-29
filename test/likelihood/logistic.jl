@testset "logistic" begin
    N, d = 20, 2
    k = 2.0 * transform(SqExponentialKernel(), 10.0)
    X, f = generate_f(N, d, k)
    y = sign.(f)
    floattypes = [Float64]
    tests_likelihood(
        LogisticLikelihood(),
        LogisticLikelihood,
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
