"""

General Framework for the data augmented Gaussian Processes

"""
module AugmentedGaussianProcesses

@enum GPModelType Undefined=0 BSVM=1 XGPC=2 Regression=3 StudentT=4 MultiClassGP=5

#Class arborescence

abstract type GPModel end

abstract type OnlineGPModel <: GPModel end

abstract type OfflineGPModel <: GPModel end

abstract type LinearModel <: OfflineGPModel end

abstract type NonLinearModel <: OfflineGPModel end

abstract type MultiClassGPModel <: OfflineGPModel end

abstract type SparseModel <: NonLinearModel end

abstract type FullBatchModel <: NonLinearModel end

export GPModel, OnlineGPModel, OfflineGPModel, SparseModel, NonLinearModel, LinearModel, FullBatchModel, GPMOdelType

# include("graddescent/GradDescent.jl")
include("kernels/KernelModule.jl")
include("kmeans/KMeansModule.jl")
include("functions/PGSampler.jl")
include("functions/PerturbativeCorrection.jl")
include("functions/GPAnalysisTools.jl")
include("functions/IO_model.jl")
#Custom modules
using .KernelModule
using .KMeansModule
using .PGSampler
using .PerturbativeCorrection
using .GPAnalysisTools
# using .IO_model
#General modules
using GradDescent
using DataFrames
using Distributions
using LinearAlgebra
using StatsBase
using SpecialFunctions
using Dates
using Expectations
using SparseArrays
using Base: show
#Exported models
export KMeansModule
export LinearBSVM, BatchBSVM, SparseBSVM
export BatchXGPC, SparseXGPC, OnlineXGPC, GibbsSamplerGPC
export BatchGPRegression, SparseGPRegression, OnlineGPRegression
export BatchStudentT, SparseStudentT
export MultiClass, SparseMultiClass
#General class definitions
#Useful functions
export getLog, getMultiClassLog
export Kernel, kerneldiagmatrix, kerneldiagmatrix!, kernelmatrix, kernelmatrix!, RBFKernel, LaplaceKernel, SigmoidKernel, PolynomialKernel, ARDKernel
export Matern3_2Kernel, Matern5_2Kernel
export fstar, multiclasspredictproba, multiclasspredictprobamcmc, multiclasspredict, ELBO
export setvalue!,getvalue,setfixed!,setfree!,getvariance,getlengthscales
export KMeansInducingPoints
# export save_trained_model,save_model,load_trained_model,load_model

global jittering = 1e-3

#using Plots

include("GPFields.jl")
include("MultiClassGPFields.jl")
#Models
include("models/LinearBSVM.jl")
include("models/BatchBSVM.jl")
include("models/SparseBSVM.jl")
include("models/BatchXGPC.jl")
include("models/SparseXGPC.jl")
include("models/OnlineXGPC.jl")
include("models/GibbsSamplerGPC.jl")
include("models/BatchGPRegression.jl")
include("models/SparseGPRegression.jl")
include("models/OnlineRegression.jl")
include("models/BatchStudentT.jl")
include("models/SparseStudentT.jl")
include("models/MultiClass.jl")
include("models/SparseMultiClass.jl")
#Functions
include("OnlineTraining.jl")
include("OfflineTraining.jl")
include("Autotuning.jl")
include("Predictions.jl")
include("models/General_Functions.jl")
include("models/BSVM_Functions.jl")
include("models/XGPC_Functions.jl")
include("models/Regression_Functions.jl")
include("models/StudentT_Functions.jl")
include("models/MultiClass_Functions.jl")

end #End Module
