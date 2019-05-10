using Distributions, LinearAlgebra
using Random: seed!
using AugmentedGaussianProcesses
using CSV, DataFrames
seed!(42)
N = 3000
D = 20
K = 4
k = RBFKernel(1.0)
X = rand(N,D)
y = rand(MvNormal(kernelmatrix(X,k)+1e-3I))
df = DataFrame(X)
df.y_reg = y
df.y_class = sign.(y)
width = maximum(y)-minimum(y);normy = (y.-minimum(y))/width*K
df.y_multi = floor.(Int64,normy)
CSV.write(joinpath(@__DIR__,"benchmarkdata.csv"),df)
