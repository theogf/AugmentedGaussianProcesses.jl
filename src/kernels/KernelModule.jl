"""
     Module for the kernel functions, also create kernel matrices
     Mostly from the list http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/
     Kernels are created by calling the constructor (available right now :
     RBFKernel,
     Arguments are kernel specific, see for each one
"""
#LaplaceKernel, SigmoidKernel, ARDKernel, PolynomialKernel, Matern3_2Kernel, Matern5_2Kernel
module KernelModule

using LinearAlgebra
using Distances
using SpecialFunctions
using GradDescent: Optimizer
include("hyperparameters/HyperParametersModule.jl")
using .HyperParametersModule:
    Bound,
        OpenBound,
        ClosedBound,
        NullBound,
    Interval,
        interval,
    HyperParameter,
    HyperParameters,
        getvalue,
        setvalue!,
        checkvalue,
        gettheta,
        checktheta,
        settheta!,
        lowerboundtheta,
        upperboundtheta,
        update!,
        setfixed!,
        setfree!,
        setparamoptimizer!

import Base: *, +, getindex, show
export Kernel, KernelSum, KernelProduct
export RBFKernel, SEKernel, LaplaceKernel, SigmoidKernel, PolynomialKernel, ARDKernel, MaternKernel, Matern3_2Kernel, Matern5_2Kernel
export kernelmatrix,kernelmatrix!,kerneldiagmatrix,kerneldiagmatrix!
export computeIndPointsJ
export apply_gradients_lengthscale!, apply_gradients_variance!, apply_gradients!
export kernelderivativematrix_K
export kernelderivativematrix,kernelderivativediagmatrix
export InnerProduct, SquaredEuclidean, Identity
export compute_hyperparameter_gradient
export compute,plotkernel
export getvalue,setvalue!,setfixed!,setfree!,setoptimizer!, getvarianceoptimizer
export getlengthscales, getvariance
export isARD,isIso
export HyperParameter,HyperParameters,Interval, OpenBound,  NullBound

# include("KernelSum.jl")
# include("KernelProduct.jl")
include("kernel.jl")
include("RBF.jl")
include("matern.jl")
include("kernelmatrix.jl")
include("kernelgradients.jl")



end #end of module
