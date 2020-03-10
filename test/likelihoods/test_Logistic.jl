include("../testingtools.jl")

N,d = 100,2
k = transform(SqExponentialKernel(),10.0)
X,f = generate_f(N,d,k)
y = sign.(f)
floattypes = [Float64]
tests_likelihood(
    "Logistic Likelihood",
    LogisticLikelihood(),
    LogisticLikelihood,
    Dict(
        "VGP" => Dict("AVI" => true, "QVI" => true, "MCVI" => false),
        "SVGP" => Dict("AVI" => true, "QVI" => true, "MCVI" => false),
        "MCGP" => Dict("Gibbs" => true, "HMC" => false),
    ),
    floattypes,
    "Classification",
    1,
)
