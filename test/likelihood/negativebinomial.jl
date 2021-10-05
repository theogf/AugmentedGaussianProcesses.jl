@testset "negativebinomial" begin
    N, d = 20, 2
    r = 10
    rfloat = 10.0
    k = SqExponentialKernel() ∘ ScaleTransform(10.0)
    X, f = generate_f(N, d, k)
    y = rand.(NegativeBinomial.(r, AGP.logistic.(f)))
    floattypes = [Float64]
    args = (
        Dict(
            "VGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => false),
            "SVGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => false),
            "OSVGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => false),
            "MCGP" => Dict("Gibbs" => true, "HMC" => false),
        ),
        floattypes,
        "NegBinomial",
        1,
        X,
        f,
        y,
        k,
    )
    tests_likelihood(NegBinomialLikelihood(r), NegBinomialLikelihood, args...)
    tests_likelihood(NegBinomialLikelihood(rfloat), NegBinomialLikelihood, args...)
end
