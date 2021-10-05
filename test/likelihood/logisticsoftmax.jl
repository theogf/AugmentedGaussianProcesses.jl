@testset "logisticsoftmax" begin
    N, d = 100, 1
    K = 3
    k = 2.0 * SqExponentialKernel() ∘ ScaleTransform(10.0)
    X, f1 = generate_f(N, d, k)
    X, f2 = generate_f(N, d, k; X=X)
    X, f3 = generate_f(N, d, k; X=X)
    f = [f1, f2, f3]
    y = getindex.(findmax(hcat(f1, f2, f3); dims=2)[2], 2)[:]
    # scatter(X,[f1,f2,f3])
    # scatter!(X,y)

    floattypes = [Float64]
    tests_likelihood(
        LogisticSoftMaxLikelihood(K),
        MultiClassLikelihood,
        Dict(
            "VGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => true),
            "SVGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => true),
            "OSVGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => false),
            "MCGP" => Dict("Gibbs" => true, "HMC" => false),
        ),
        floattypes,
        "MultiClass",
        K,
        X,
        f,
        y,
        k,
    )
end
