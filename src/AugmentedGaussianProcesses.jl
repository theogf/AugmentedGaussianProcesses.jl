"""

General Framework for the data augmented Gaussian Processes

"""
module AugmentedGaussianProcesses

const AGP = AugmentedGaussianProcesses
export AGP
export AbstractGP, GP, VGP, SVGP, VStP, MCGP, MOVGP, MOSVGP, MOARGP, OnlineSVGP # All models
export Likelihood,  RegressionLikelihood, ClassificationLikelihood, MultiClassLikelihood, EventLikelihood # All categories of likelihoods
export GaussianLikelihood, StudentTLikelihood, LaplaceLikelihood, HeteroscedasticLikelihood # Regression Likelihoods
export LogisticLikelihood, BayesianSVM # Classification Likelihoods
export SoftMaxLikelihood, LogisticSoftMaxLikelihood # Multiclass Classification Likelihoods
export PoissonLikelihood, NegBinomialLikelihood # Event Likelihoods
export Inference, Analytic, AnalyticVI, AnalyticSVI # Inference objects
export GibbsSampling, HMCSampling # Sampling inference
export NumericalVI, NumericalSVI, MCIntegrationVI, MCIntegrationSVI, QuadratureVI, QuadratureSVI # Numerical inference
export PriorMean, ZeroMean, ConstantMean, EmpiricalMean, AffineMean # Prior means
#Useful functions
export train!, sample
export predict_f, predict_y, proba_y
export fstar
export ELBO
export covariance, diag_covariance, prior_mean
export @augmodel

#General modules
using Reexport
using LinearAlgebra
using Random
@reexport using KernelFunctions
using KernelFunctions: ColVecs, RowVecs
using Zygote, ForwardDiff
using ChainRulesCore: ChainRulesCore, NO_FIELDS
using Flux: params, destructure
@reexport using Flux.Optimise
using AdvancedHMC
using MCMCChains
using StatsBase
using StatsFuns
using SpecialFunctions
using Distributions: 
                Distributions, Distribution,
                dim, cov, mean, var, 
                pdf, logpdf, loglikelihood,
                Normal, Poisson, NegativeBinomial, InverseGamma, Laplace, MvNormal, Gamma
using FastGaussQuadrature: gausshermite
using ProgressMeter, SimpleTraits

#Useful functions and module
include(joinpath("functions", "PGSampler.jl"))
include(joinpath("functions", "GIGSampler.jl"))
include(joinpath("functions", "lap_transf_dist.jl"))
#include("functions/PerturbativeCorrection.jl")
# include("functions/GPAnalysisTools.jl")
# include("functions/IO_model.jl")
#Custom modules
using .PGSampler
using .GIGSampler

include(joinpath("inducingpoints" , "InducingPoints.jl"))
@reexport using .InducingPoints

# using .PerturbativeCorrection
# using .GPAnalysisTools
# using .IO_model


# Main classes
abstract type Inference{T<:Real} end
abstract type VariationalInference{T} <: Inference{T} end
abstract type SamplingInference{T} <: Inference{T} end
abstract type Likelihood{T<:Real} end
abstract type AbstractLatent{T<:Real,Tpr,Tpo} end

include(joinpath("mean", "priormean.jl"))
include(joinpath("data", "datacontainer.jl"))
include(joinpath("functions", "utils.jl"))

# Models
include(joinpath("models", "AbstractGP.jl"))
include(joinpath("gpblocks", "latentgp.jl"))
include(joinpath("models", "GP.jl"))
include(joinpath("models", "VGP.jl"))
include(joinpath("models", "MCGP.jl"))
include(joinpath("models", "SVGP.jl"))
include(joinpath("models", "VStP.jl"))
include(joinpath("models", "MOSVGP.jl"))
include(joinpath("models", "MOVGP.jl"))
include(joinpath("models", "OnlineSVGP.jl"))
include(joinpath("models", "single_output_utils.jl"))
include(joinpath("models", "multi_output_utils.jl"))

include(joinpath("inference", "inference.jl"))
include(joinpath("likelihood", "likelihood.jl"))
include(joinpath("likelihood", "generic_likelihood.jl"))

include(joinpath("functions", "KLdivergences.jl"))
include(joinpath("functions", "ELBO.jl"))
include(joinpath("data", "utils.jl"))
include(joinpath("functions", "plotting.jl"))

# Training and prediction functions
include(joinpath("training", "training.jl"))
include(joinpath("training", "onlinetraining.jl"))
include(joinpath("hyperparameter", "autotuning.jl"))
include(joinpath("training", "predictions.jl"))
include("ar_predict.jl")

end #End Module
