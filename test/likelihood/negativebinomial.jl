N,d = 100,2
r = 10
k = transform(SqExponentialKernel(),10.0)
X,f = generate_f(N,d,k)
y = rand.(NegativeBinomial.(r, AGP.logistic.(f)))
floattypes = [Float64]
tests_likelihood(
    "Negative Binomial Likelihood",
    NegBinomialLikelihood(r),
    NegBinomialLikelihood,
    Dict(
        "VGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => false),
        "SVGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => false),
        "OSVGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => false),
        "MCGP" => Dict("Gibbs" => true, "HMC" => false),
    ),
    floattypes,
    "NegBinomial",
    1,
)
