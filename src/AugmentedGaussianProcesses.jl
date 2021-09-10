"""

General Framework for the data augmented Gaussian Processes

"""
module AugmentedGaussianProcesses

const AGP = AugmentedGaussianProcesses
export AGP
export AbstractGPModel, GP, VGP, SVGP, VStP, MCGP, MOVGP, MOSVGP, MOARGP, OnlineSVGP # All models
export AbstractLikelihood,
    RegressionLikelihood, ClassificationLikelihood, MultiClassLikelihood, EventLikelihood # All categories of likelihoods
export GaussianLikelihood, StudentTLikelihood, LaplaceLikelihood, HeteroscedasticLikelihood # Regression Likelihoods
export LogisticLikelihood, BayesianSVM # Classification Likelihoods
export SoftMaxLikelihood, LogisticSoftMaxLikelihood # Multiclass Classification Likelihoods
export PoissonLikelihood, NegBinomialLikelihood # Event Likelihoods
export AbstractInference, Analytic, AnalyticVI, AnalyticSVI # Inference objects
export GibbsSampling, HMCSampling # Sampling inference
export NumericalVI,
    NumericalSVI, MCIntegrationVI, MCIntegrationSVI, QuadratureVI, QuadratureSVI # Numerical inference
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
@reexport using KernelFunctions
@reexport using Optimisers
@reexport using InducingPoints

using AbstractMCMC
using AdvancedHMC
using ChainRulesCore: ChainRulesCore, NoTangent
using Distributions:
    Distributions,
    Distribution,
    dim,
    cov,
    mean,
    var,
    pdf,
    logpdf,
    loglikelihood,
    Normal,
    Poisson,
    NegativeBinomial,
    InverseGamma,
    Laplace,
    MvNormal,
    Gamma
using FastGaussQuadrature: gausshermite
using ForwardDiff
using KernelFunctions: ColVecs, RowVecs
using LinearAlgebra
using ProgressMeter
using Random
using StatsBase
using SimpleTraits
using StatsFuns
using SpecialFunctions
using Zygote

#Include custom module for additional distributions
include(joinpath("ComplementaryDistributions", "ComplementaryDistributions.jl"))
using .ComplementaryDistributions

# Main classes
abstract type AbstractInference{T<:Real} end
abstract type VariationalInference{T} <: AbstractInference{T} end
abstract type SamplingInference{T} <: AbstractInference{T} end
abstract type AbstractLikelihood{T<:Real} end
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
include(joinpath("training", "sampling.jl"))
include(joinpath("training", "onlinetraining.jl"))
include(joinpath("hyperparameter", "autotuning.jl"))
include(joinpath("training", "predictions.jl"))
include("ar_predict.jl")

end #End Module
