"""

General Framework for the data augmented Gaussian Processes

"""
module AugmentedGaussianProcesses

export AbstractGP, GP, VGP, SVGP, VStP
export Likelihood,  RegressionLikelihood, ClassificationLikelihood, MultiClassLikelihood, EventLikelihood
export GaussianLikelihood, StudentTLikelihood, LaplaceLikelihood, HeteroscedasticLikelihood, Matern3_2Likelihood, MaternLikelihood
export LogisticLikelihood, BayesianSVM
export SoftMaxLikelihood, LogisticSoftMaxLikelihood
export PoissonLikelihood
export Inference, Analytic, AnalyticVI, AnalyticSVI, GibbsSampling, MCIntegrationVI, MCIntegrationSVI, QuadratureVI, QuadratureSVI
export NumericalVI, NumericalSVI
export PriorMean, ZeroMean, ConstantMean, EmpiricalMean
#Deprecated
export BatchGPRegression, SparseGPRegression, MultiClass, SparseMultiClass, BatchBSVM, SparseBSVM, BatchXGPC, SparseXGPC, BatchStudentT, SparseStudentT


#Useful functions and module
include("kernels/KernelModule.jl")
include("kmeans/KMeansModule.jl")
include("functions/PGSampler.jl")
include("functions/GIGSampler.jl")
#include("functions/PerturbativeCorrection.jl")
# include("functions/GPAnalysisTools.jl")
# include("functions/IO_model.jl")
#Custom modules
using .KernelModule
using .KMeansModule
using .PGSampler
using .GIGSampler
# using .PerturbativeCorrection
# using .GPAnalysisTools
# using .IO_model
#General modules
# using MLKernels
using GradDescent
export Optimizer, Adam, VanillaGradDescent, ALRSVI, InverseDecay
using DataFrames, LinearAlgebra
using StatsBase, StatsFuns, SpecialFunctions, Expectations, Random, Distributions, FastGaussQuadrature
using ProgressMeter
import Base: convert, show, copy
#Exported modules
export KMeansModule
#Useful functions
export train!
export predict_f, predict_y, proba_y
# export getLog, getMultiClassLog
export Kernel,  Matern3_2Kernel, Matern5_2Kernel, RBFKernel, LaplaceKernel, SigmoidKernel, PolynomialKernel, ARDKernel
export kerneldiagmatrix, kerneldiagmatrix!, kernelmatrix, kernelmatrix!
export fstar, multiclasspredictproba, multiclasspredictprobamcmc, multiclasspredict, ELBO
export setvalue!,getvalue,setfixed!,setfree!,getvariance,getlengthscales,setoptimizer!
export opt_diag, opt_trace
export covariance, diag_covariance, prior_mean
export KMeansInducingPoints

# Main classes
abstract type Inference{T<:Real} end
abstract type Likelihood{T<:Real}  end

const LatentArray = Vector #For future optimization : How collection of latent GP parameters and local variables are stored
include("prior/priormean.jl")

include("models/AbstractGP.jl")
include("models/GP.jl")
include("models/VGP.jl")
include("models/SVGP.jl")
include("models/VStP.jl")

include("inference/inference.jl")
include("likelihood/likelihood.jl")

include("functions/utils.jl")
include("functions/init.jl")
include("functions/KLdivergences.jl")
# include("functions/plotting.jl")
#Deprecated constructors
include("deprecated.jl")

#Training Functions
include("training.jl")
include("autotuning.jl")
include("predictions.jl")
end #End Module
