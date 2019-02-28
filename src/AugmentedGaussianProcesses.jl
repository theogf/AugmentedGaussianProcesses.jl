"""

General Framework for the data augmented Gaussian Processes

"""
module AugmentedGaussianProcesses

export GP, VGP, SVGP
export Likelihood, GaussianLikelihood, LogisticLikelihood
export MultiClassLikelihood, SoftMaxLikelihood, LogisticSoftMaxLikelihood
export AugmentedLogisticSoftMaxLikelihood
export Inference, AnalyticInference, StochasticAnalyticInference, GibbsSampling, MCMCIntegrationInference, QuadratureInference
export NumericalInference, StochasticNumericalInference
export ALRSVI

#Deprecated
export MultiClass, SparseMultiClass


#Useful functions and module
include("kernels/KernelModule.jl")
include("kmeans/KMeansModule.jl")
include("functions/PGSampler.jl")
include("functions/PerturbativeCorrection.jl")
include("functions/GPAnalysisTools.jl")
# include("functions/IO_model.jl")
#Custom modules
using .KernelModule
using .KMeansModule
using .PGSampler
using .PerturbativeCorrection
using .GPAnalysisTools
# using .IO_model
#General modules
# using MLKernels
using GradDescent
import GradDescent: update
using DataFrames
using Distributions
using LinearAlgebra
using StatsBase
using StatsFuns
using SpecialFunctions
using Dates
using Expectations
using SparseArrays
import Base: convert, show, copy
#Exported models
export KMeansModule
#Useful functions
export train!
export predict_f, predict_y, proba_y
export getLog, getMultiClassLog
export Kernel,  Matern3_2Kernel, Matern5_2Kernel, RBFKernel, LaplaceKernel, SigmoidKernel, PolynomialKernel, ARDKernel
export kerneldiagmatrix, kerneldiagmatrix!, kernelmatrix, kernelmatrix!
export fstar, multiclasspredictproba, multiclasspredictprobamcmc, multiclasspredict, ELBO
export setvalue!,getvalue,setfixed!,setfree!,getvariance,getlengthscales,setoptimizer!
export opt_diag, opt_trace
export KMeansInducingPoints

# Main classes
abstract type Inference{T<:Real} end
abstract type Likelihood{T<:Real}  end

include("models/GP.jl")
include("models/VGP.jl")
include("models/SVGP.jl")

include("inference/inference.jl")
include("likelihood/likelihood.jl")

include("functions/utils.jl")
include("functions/init.jl")
include("functions/KLdivergences.jl")
include("functions/alrsvi.jl")
include("functions/inversedecay.jl")
#Deprecated constructors
include("deprecated.jl")

#Training Functions
include("training.jl")
include("autotuning.jl")
include("predictions.jl")
# include("models/General_Functions.jl")

end #End Module
