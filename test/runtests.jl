using OMGP
using Test

# write your own tests here
@test include("test_XGPC.jl")
@test include("test_BSVM.jl")
@test include("test_Regression.jl")
@test include("test_StudentT.jl")
