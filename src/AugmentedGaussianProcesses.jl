"""

General Framework for the data augmented Gaussian Processes

"""
module AugmentedGaussianProcesses
    const AGP = AugmentedGaussianProcesses; export AGP
    export AbstractGP, GP, VGP, SVGP, VStP, MCGP, MOVGP, MOSVGP, MOARGP,OnlineSVGP
    export Likelihood,  RegressionLikelihood, ClassificationLikelihood, MultiClassLikelihood
    export GaussianLikelihood, StudentTLikelihood, LaplaceLikelihood, HeteroscedasticLikelihood
    export LogisticLikelihood, BayesianSVM
    export SoftMaxLikelihood, LogisticSoftMaxLikelihood
    export PoissonLikelihood, NegBinomialLikelihood
    export Inference, Analytic, AnalyticVI, AnalyticSVI, GibbsSampling, HMCSampling, MCIntegrationVI, MCIntegrationSVI, QuadratureVI, QuadratureSVI
    export NumericalVI, NumericalSVI
    export PriorMean, ZeroMean, ConstantMean, EmpiricalMean, AffineMean
    #Useful functions
    export train!, sample
    export predict_f, predict_y, proba_y
    export fstar, ELBO
    export covariance, diag_covariance, prior_mean

    #General modules
    using Reexport
    using LinearAlgebra
    using Random
    @reexport using KernelFunctions
    using KernelFunctions: ColVecs, RowVecs
    @reexport using InducingPoints
    using Zygote
    using ForwardDiff
    using Flux #Remove full dependency on Flux once params for KernelFunctions is set
    @reexport using Flux.Optimise
    using PDMats: PDMat, invquad
    using AdvancedHMC
    using MCMCChains
    using StatsBase
    using StatsFuns
    using SpecialFunctions
    using Distributions
    using FastGaussQuadrature: gausshermite
    using ProgressMeter, SimpleTraits
    #Exported modules
    # export KMeansModule
    export KMeansInducingPoints

    #Useful functions and module
    include("functions/PGSampler.jl")
    include("functions/GIGSampler.jl")
    include("functions/lap_transf_dist.jl")
    #include("functions/PerturbativeCorrection.jl")
    # include("functions/GPAnalysisTools.jl")
    # include("functions/IO_model.jl")
    #Custom modules
    @reexport using .IPModule
    using .PGSampler
    using .GIGSampler
    # using .PerturbativeCorrection
    # using .GPAnalysisTools
    # using .IO_model

    # Main classes
    abstract type Inference{T<:Real} end
    abstract type VariationalInference{T} <: Inference{T} end
    abstract type SamplingInference{T} <: Inference{T} end
    abstract type Likelihood{T<:Real} end
    abstract type Abstract_GP{T<:Real, K<:Kernel, Tmean<:PriorMean} end

    include("functions/utils.jl")
    include("prior/priormean.jl")

    # Models
    include("models/AbstractGP.jl")
    include("models/GP_base.jl")
    include("models/GP.jl")
    include("models/VGP.jl")
    include("models/MCGP.jl")
    include("models/SVGP.jl")
    include("models/VStP.jl")
    include("models/MOSVGP.jl")
    include("models/MOARGP.jl")
    include("models/MOVGP.jl")
    include("models/OnlineSVGP.jl")
    include("models/single_output_utils.jl")
    include("models/multi_output_utils.jl")

    include("inference/inference.jl")
    include("likelihood/likelihood.jl")

    include("functions/init.jl")
    include("functions/KLdivergences.jl")
    include("functions/plotting.jl")

    #Training Functions
    include("training.jl")
    include("onlinetraining.jl")
    include("hyperparameter/autotuning.jl")
    include("predictions.jl")
    include("ar_predict.jl")
end #End Module
