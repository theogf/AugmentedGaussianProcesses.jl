using AugmentedGaussianProcesses
using Test
using Random: seed!
seed!(42)

# Global flags for the tests
doPlots = false
verbose = 0
@test include("test_XGPC.jl")
@test include("test_BSVM.jl")
@test include("test_Regression.jl")
@test include("test_StudentT.jl")
@test include("test_IO.jl")
