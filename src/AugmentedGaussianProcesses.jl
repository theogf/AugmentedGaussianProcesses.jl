"""

General Framework for the data augmented Gaussian Processes

"""
module AugmentedGaussianProcesses

    export AbstractGP, GP, VGP, SVGP, VStP, MCGP, MOSVGP, OnlineSVGP
    export Likelihood,  RegressionLikelihood, ClassificationLikelihood, MultiClassLikelihood
    export GaussianLikelihood, StudentTLikelihood, LaplaceLikelihood, HeteroscedasticLikelihood
    export LogisticLikelihood, BayesianSVM
    export SoftMaxLikelihood, LogisticSoftMaxLikelihood
    export PoissonLikelihood, NegBinomialLikelihood
    export Inference, Analytic, AnalyticVI, AnalyticSVI, GibbsSampling, MCIntegrationVI, MCIntegrationSVI, QuadratureVI, QuadratureSVI
    export NumericalVI, NumericalSVI
    export PriorMean, ZeroMean, ConstantMean, EmpiricalMean, AffineMean

    #Useful functions and module
    include("kmeans/KMeansModule.jl")
    include("functions/PGSampler.jl")
    include("functions/GIGSampler.jl")
    #include("functions/PerturbativeCorrection.jl")
    # include("functions/GPAnalysisTools.jl")
    # include("functions/IO_model.jl")
    #Custom modules
    # using .KMeansModule
    using .PGSampler
    using .GIGSampler
    # using .PerturbativeCorrection
    # using .GPAnalysisTools
    # using .IO_model
    #General modules
    using Reexport
    using LinearAlgebra, Random
    @reexport using KernelFunctions
    using Zygote, ForwardDiff
    using Flux: functorm, params
    @reexport using Flux.Optimise
    using PDMats
    using AdvancedHMC, MCMCChains
    using StatsBase, StatsFuns, SpecialFunctions, Distributions, FastGaussQuadrature
    using ProgressMeter, SimpleTraits
    #Exported modules
    # export KMeansModule
    export KMeansInducingPoints
    #Useful functions
    export train!, sample
    export predict_f, predict_y, proba_y
    export fstar, ELBO
    export covariance, diag_covariance, prior_mean

    # Main classes
    abstract type Inference{T<:Real} end
    abstract type VariationalInference{T} <: Inference{T} end
    abstract type SamplingInference{T} <: Inference{T} end
    abstract type Likelihood{T<:Real} end
    abstract type Abstract_GP{T<:Real} end

    const LatentArray = Vector #For future optimization : How collection of latent GP parameters and local variables are stored
    include("functions/utils.jl")
    include("prior/priormean.jl")
    # include("models/inducing_points.jl")

    include("models/AbstractGP.jl")
    include("models/GP_base.jl")
    include("models/GP.jl")
    include("models/VGP.jl")
    include("models/MCGP.jl")
    include("models/SVGP.jl")
    include("models/VStP.jl")
    include("models/MOSVGP.jl")
    include("models/OnlineSVGP.jl")

    include("inference/inference.jl")
    include("likelihood/likelihood.jl")

    include("functions/init.jl")
    include("functions/KLdivergences.jl")
    include("functions/plotting.jl")

    #Training Functions
    include("training.jl")
    include("onlinetraining.jl")
    include("autotuning.jl")
    include("predictions.jl")
end #End Module
