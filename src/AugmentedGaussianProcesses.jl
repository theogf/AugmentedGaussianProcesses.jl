"""

General Framework for the data augmented Gaussian Processes

"""
module AugmentedGaussianProcesses

export GP, VGP, SVGP
export Likelihood, GaussianLikelihood
export Inference, AnalyticInference, GibbsSampling



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
using DataFrames
using Distributions
using LinearAlgebra
using StatsBase
using SpecialFunctions
using Dates
using Expectations
using SparseArrays
import Base: convert, show
#Exported models
export KMeansModule
#Useful functions
export train!
export predit_f, predict_y
export getLog, getMultiClassLog
export Kernel,  Matern3_2Kernel, Matern5_2Kernel, RBFKernel, LaplaceKernel, SigmoidKernel, PolynomialKernel, ARDKernel
export kerneldiagmatrix, kerneldiagmatrix!, kernelmatrix, kernelmatrix!
export fstar, multiclasspredictproba, multiclasspredictprobamcmc, multiclasspredict, ELBO
export setvalue!,getvalue,setfixed!,setfree!,getvariance,getlengthscales
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
#Training Functions
include("training.jl")
include("autotuning.jl")
include("predictions.jl")
# include("models/General_Functions.jl")

end #End Module
