include("../testingtools.jl")

N,d = 100,2
k = transform(SqExponentialKernel(),10.0)
X,f = generate_f(N,d,k)
y = sign.(f)
floattypes = [Float64]
tests_likelihood(
    "Bayesian SVM",
    BayesianSVM(),
    BayesianSVM,
    Dict(
        "VGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => false),
        "SVGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => false),
        "OSVGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => false),
        "MCGP" => Dict("Gibbs" => false, "HMC" => false),
    ),
    floattypes,
    "Classification",
    1,
)
