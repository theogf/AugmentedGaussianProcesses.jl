using AugmentedGaussianProcesses.KernelModule
# using MLKernels
using Random: seed!
using ForwardDiff
seed!(42)
A = rand(100,100)
## TODO Compare kernel results with MLKernels package
function make_matrix(sigma)
    k = RBFKernel(sigma[1])
    kernelmatrix(A,k)
end
## Compute matrix variations via finite differences and compare with analytical
ForwardDiff.gradient(make_matrix,[0.5])
