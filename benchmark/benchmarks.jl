using BenchmarkTools
using Random
using AugmentedGaussianProcesses

const SUITE = BenchmarkGroup()

# SUITE["Kernel"] = BenchmarkGroup()
# include("kernel.jl")
SUITE["Models"] = BenchmarkGroup()
include("models.jl")
