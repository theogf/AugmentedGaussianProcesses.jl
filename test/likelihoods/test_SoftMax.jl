using AugmentedGaussianProcesses
using Test


include("../testingtools.jl")

N,d = 100,1
K = 3
k = transform(SqExponentialKernel(),10.0)
X,f1 = generate_f(N,d,k)
X,f2 = generate_f(N,d,k;X=X)
X,f3 = generate_f(N,d,k;X=X)
f = [f1,f2,f3]
y = getindex.(findmax(hcat(f1,f2,f3),dims=2)[2],2)[:]
# scatter(X,[f1,f2,f3])
# scatter!(X,y)

floattypes = [Float64]
tests_likelihood(
    "Soft-Max Likelihood",
    SoftMaxLikelihood(K),
    SoftMaxLikelihood,
    Dict(
        "VGP" => Dict("AVI" => false, "QVI" => false, "MCVI" => true),
        "SVGP" => Dict("AVI" => false, "QVI" => false, "MCVI" => true),
        "MCGP" => Dict("Gibbs" => false, "HMC" => false),
    ),
    floattypes,
    "MultiClass",
    K,
)
