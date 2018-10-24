using OMGP
using Test

# write your own tests here
doPlots = false
@test include("test_XGPC.jl")
@test include("test_BSVM.jl")
@test include("test_Regression.jl")
@test include("test_StudentT.jl")
@test include("test_IO.jl")
