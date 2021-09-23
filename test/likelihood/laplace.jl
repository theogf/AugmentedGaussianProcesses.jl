@testset "laplace" begin
    N, d = 20, 2
    β = 3.0
    k = SqExponentialKernel() ∘ ScaleTransform(10.0)
    X, f = generate_f(N, d, k)
    y = f + rand(Laplace(β), N)
    floattypes = [Float64]
    tests_likelihood(
        LaplaceLikelihood(β),
        LaplaceLikelihood,
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
