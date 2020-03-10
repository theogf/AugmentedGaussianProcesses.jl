include("../testingtools.jl")

N,d = 100,2
λ = 5.0
k = transform(SqExponentialKernel(),10.0)
X,f = generate_f(N,d,k)
y = rand.(Poisson.(λ*AGP.logistic.(f)))
floattypes = [Float64]
tests_likelihood(
    "Poisson Likelihood",
    PoissonLikelihood(λ),
    PoissonLikelihood,
    Dict(
        "VGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => false),
        "SVGP" => Dict("AVI" => true, "QVI" => false, "MCVI" => false),
        "MCGP" => Dict("Gibbs" => false, "HMC" => false),
    ),
    floattypes,
    "Poisson",
    1,
)
