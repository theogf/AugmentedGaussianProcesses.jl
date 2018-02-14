###########################

#General Framework for the data augmented Gaussian Processes

###########################
module OMGP

include("KernelFunctions.jl")
include("KMeansModule.jl")
include("PGSampler.jl")
include("PerturbativeCorrection.jl")
#Custom modules
using .KernelFunctions
using .KMeansModule
using .PGSampler
using .PerturbativeCorrection
#General modules
using Distributions
using StatsBase
using QuadGK
using GradDescent
using ValueHistories

#Exported models
export LinearBSVM, BatchBSVM, SparseBSVM
export BatchXGPC, SparseXGPC, GibbsSamplerGPC
#General class definitions
export AugmentedModel, SparseModel, NonLinearModel, LinearModel, FullBatchModel
export ELBO

@enum ClassifierType Undefined=0 BSVM=1 XGPC=2

#Class arborescence

abstract type AugmentedModel end

abstract type LinearModel <: AugmentedModel end

abstract type NonLinearModel <: AugmentedModel end

abstract type SparseModel <: NonLinearModel end

abstract type FullBatchModel <: NonLinearModel end

include("GPFields.jl")
#Models
include("LinearBSVM.jl")
include("BatchBSVM.jl")
include("SparseBSVM.jl")
include("BatchXGPC.jl")
include("SparseXGPC.jl")
include("GibbsSamplerGPC.jl")
#Functions
include("ModelFunctions.jl")
include("GPFunctions.jl")
#include("BSVM_Functions.jl")
include("XGPC_Functions.jl")

end #End Module
