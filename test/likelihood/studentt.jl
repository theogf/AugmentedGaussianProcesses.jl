@testset "studentt" begin
    N, d = 20, 2
    ν = 3.0
    k = SqExponentialKernel() ∘ ScaleTransform(10.0)
    X, f = generate_f(N, d, k)
    y = f + rand(TDist(ν), N)
    floattypes = [Float64]
    tests_likelihood(
        StudentTLikelihood(ν),
        StudentTLikelihood,
        Dict(
            "VGP" => Dict("AVI" => true, "QVI" => true, "MCVI" => false),
            "SVGP" => Dict("AVI" => true, "QVI" => true, "MCVI" => false),
            "OSVGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => false),
            "MCGP" => Dict("Gibbs" => true, "HMC" => false),
        ),
        floattypes,
        "Regression",
        1,
        X,
        f,
        y,
        k,
    )
end
