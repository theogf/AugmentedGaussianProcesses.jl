using AugmentedGaussianProcesses
using Test

include("../testingtools.jl")

N,d = 100,2
β = 3.0
k = transform(SqExponentialKernel(),10.0)
X,f = generate_f(N,d,k)
y = f + rand(Laplace(β),N)
floattypes = [Float64]
tests_likelihood(
    "Laplace Likelihood",
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
)
