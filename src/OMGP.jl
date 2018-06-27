"""

General Framework for the data augmented Gaussian Processes

"""
module OMGP

@enum GPModelType Undefined=0 BSVM=1 XGPC=2 Regression=3 MultiClassGP=4 EPGPC=5

#Class arborescence

abstract type GPModel end

abstract type LinearModel <: GPModel end

abstract type NonLinearModel <: GPModel end

abstract type MultiClassGPModel <: GPModel end

abstract type SparseModel <: NonLinearModel end

abstract type FullBatchModel <: NonLinearModel end


include("KernelFunctions.jl")
include("KMeansModule.jl")
include("PGSampler.jl")
include("PerturbativeCorrection.jl")
#Custom modules
using .KernelFunctions
using .KMeansModule
using .PGSampler
using .PerturbativeCorrection
include("GPAnalysisTools.jl")
using .GPAnalysisTools
#General modules
using Distributions
using StatsBase
using QuadGK
using GradDescent
using ValueHistories
using Gallium #TODO remove for published version
#Exported models
export LinearBSVM, BatchBSVM, SparseBSVM
export BatchXGPC, SparseXGPC, GibbsSamplerGPC, EPSparseGPC
export GPRegression, SparseGPRegression
export MultiClass, SparseMultiClass
#General class definitions
export GPModel, SparseModel, NonLinearModel, LinearModel, FullBatchModel
#Useful functions
export getLog, getMultiClassLog
export Kernel, diagkernelmatrix, kernelmatrix, RBFKernel, LaplaceKernel, SigmoidKernel, PolynomialKernel, ARDKernel
export fstar, multiclasspredictproba, multiclasspredictprobamcmc, multiclasspredict, ELBO
export setvalue!,getvalue,setfixed!,setfree!
export KMeansInducingPoints


using Plots

include("GPFields.jl")
include("MultiClassGPFields.jl")
#Models
include("models/LinearBSVM.jl")
include("models/BatchBSVM.jl")
include("models/SparseBSVM.jl")
include("models/BatchXGPC.jl")
include("models/SparseXGPC.jl")
include("models/GibbsSamplerGPC.jl")
include("models/Regression.jl")
include("models/SparseRegression.jl")
include("models/MultiClass.jl")
include("models/SparseMultiClass.jl")
include("models/SparseEPGPC.jl")
#Functions
include("Training.jl")
include("Autotuning.jl")
include("Predictions.jl")
include("models/BSVM_Functions.jl")
include("models/XGPC_Functions.jl")
include("models/Reg_Functions.jl")
include("models/MultiClass_Functions.jl")
include("models/EP_Functions.jl")

end #End Module
