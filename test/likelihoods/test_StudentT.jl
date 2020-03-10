include("../testingtools.jl")

N,d = 100,2
ν = 3.0
k = transform(SqExponentialKernel(),10.0)
X,f = generate_f(N,d,k)
y = f + rand(TDist(ν),N)
floattypes = [Float64]
tests_likelihood(
    "Student-T Likelihood",
    StudentTLikelihood(ν),
    StudentTLikelihood,
    Dict(
        "VGP" => Dict("AVI" => true, "QVI" => true, "MCVI" => false),
        "SVGP" => Dict("AVI" => true, "QVI" => true, "MCVI" => false),
        "MCGP" => Dict("Gibbs" => true, "HMC" => false),
    ),
    floattypes,
    "Regression",
    1,
)
