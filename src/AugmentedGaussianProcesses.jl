"""

General Framework for the data augmented Gaussian Processes

"""
module AugmentedGaussianProcesses

const AGP = AugmentedGaussianProcesses
export AGP
export AbstractGPModel, GP, VGP, SVGP, VStP, MCGP, MOVGP, MOSVGP, MOARGP, OnlineSVGP # All models
export AbstractLikelihood,
    RegressionLikelihood, BernoulliLikelihood, MultiClassLikelihood # All categories of likelihoods
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

using AbstractMCMC: AbstractMCMC, step, sample
# using AdvancedHMC
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
using GPLikelihoods: GPLikelihoods, AbstractLink
using GPLikelihoods: BernoulliLikelihood, PoissonLikelihood, HeteroscedasticGaussianLikelihood
using GPLikelihoods: LogisticLink, SoftMaxLink
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
include("ComplementaryDistributions/ComplementaryDistributions.jl")
using .ComplementaryDistributions

# Main classes
abstract type AbstractInference{T<:Real} end
abstract type VariationalInference{T} <: AbstractInference{T} end
abstract type SamplingInference{T} <: AbstractInference{T} end
abstract type AbstractLikelihood end
abstract type AbstractLatent{T<:Real,Tpr,Tpo} end

include("mean/priormean.jl")
include("data/datacontainer.jl")
include("functions/utils.jl")

# Models
include("models/AbstractGP.jl")
include("gpblocks/latentgp.jl")
include("models/GP.jl")
include("models/VGP.jl")
include("models/MCGP.jl")
include("models/SVGP.jl")
include("models/VStP.jl")
include("models/MOSVGP.jl")
include("models/MOVGP.jl")
include("models/OnlineSVGP.jl")
include("models/single_and_multi_output_utils.jl")

include("inference/inference.jl")
include("likelihood/likelihood.jl")
include("likelihood/generic_likelihood.jl")

include("functions/KLdivergences.jl")
include("functions/ELBO.jl")
include("data/utils.jl")
include("functions/plotting.jl")

# Training and prediction functions
include("training/states.jl")
include("training/training.jl")
include("training/sampling.jl")
include("training/onlinetraining.jl")
include("hyperparameter/autotuning.jl")
include("training/predictions.jl")
include("ar_predict.jl")

end #End Module
