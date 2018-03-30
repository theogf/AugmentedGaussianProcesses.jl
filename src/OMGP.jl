"""

General Framework for the data augmented Gaussian Processes

"""
module OMGP

@enum GPModelType Undefined=0 BSVM=1 XGPC=2 Regression=3 MultiClass=4

#Class arborescence

abstract type GPModel end

abstract type LinearModel <: GPModel end

abstract type NonLinearModel <: GPModel end

abstract type SparseModel <: NonLinearModel end

abstract type FullBatchModel <: NonLinearModel end

include("KernelFunctions.jl")
include("KMeansModule.jl")
include("PGSampler.jl")
include("PerturbativeCorrection.jl")
include("GPAnalysisTools.jl")
#Custom modules
using .KernelFunctions
using .KMeansModule
using .PGSampler
using .PerturbativeCorrection
using .GPAnalysisTools
#General modules
using Distributions
using StatsBase
using QuadGK
using GradDescent
using ValueHistories
using Gallium
#Exported models
export LinearBSVM, BatchBSVM, SparseBSVM
export BatchXGPC, SparseXGPC, GibbsSamplerGPC
export GPRegression, SparseGPRegression
#General class definitions
export GPModel, SparseModel, NonLinearModel, LinearModel, FullBatchModel
#Useful functions
export getLog
export Kernel, diagkernelmatrix, kernelmatrix, RBFKernel, LaplaceKernel, SigmoidKernel, PolynomialKernel, ARDKernel

include("GPFields.jl")
#Models
include("LinearBSVM.jl")
include("BatchBSVM.jl")
include("SparseBSVM.jl")
include("BatchXGPC.jl")
include("SparseXGPC.jl")
include("GibbsSamplerGPC.jl")
include("Regression.jl")
include("SparseRegression.jl")
include("MultiClass.jl")
#Functions
include("Training.jl")
include("Autotuning.jl")
include("Predictions.jl")
include("BSVM_Functions.jl")
include("XGPC_Functions.jl")
include("Reg_Functions.jl")

end #End Module
