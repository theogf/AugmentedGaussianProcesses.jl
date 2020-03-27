""" Class for sparse variational Gaussian Processes """
mutable struct OnlineSVGP{T<:Real,TLikelihood<:Likelihood{T},TInference<:Inference{T},N} <: AbstractGP{T,TLikelihood,TInference,N}
    X::Matrix{T} #Feature vectors
    y::Vector #Output (-1,1 for classification, real for regression, matrix for multiclass)
    nDim::Int64 # Number of covariates per data point
    nFeatures::Int64 # Number of features of the GP (equal to number of points)
    nLatent::Int64 # Number of latent GPs
    f::NTuple{N,_OSVGP}
    likelihood::TLikelihood
    inference::TInference
    verbose::Int64
    atfrequency::Int64
    Trained::Bool
end

"""Create a sparse variational Gaussian Process model
Argument list :

**Mandatory arguments**
 # - `X` : input features, should be a matrix N×D where N is the number of observation and D the number of dimension
 # - `y` : input labels, can be either a vector of labels for multiclass and single output or a matrix for multi-outputs (note that only one likelihood can be applied)
 - `kernel` : covariance function, can be either a single kernel or a collection of kernels for multiclass and multi-outputs models
 - `likelihood` : likelihood of the model, currently implemented : Gaussian, Bernoulli (with logistic link), Multiclass (softmax or logistic-softmax) see [`Likelihood`](@ref)
 - `inference` : inference for the model, can be analytic, numerical or by sampling, check the model documentation to know what is available for your likelihood see [`Inference`](@ref)
 - `ZAlg` : Algorithm to add automatically inducing points, `CircleKMeans` by default, options are : `OfflineKMeans`, `StreamingKMeans`, `Webscale`
 - `nLatent` : Number of needed latent `f`
**Optional arguments**
 - `verbose` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
 - `Autotuning` : Flag for optimizing hyperparameters
 - `atfrequency` : Choose how many variational parameters iterations are between hyperparameters optimization
 - `optimiser` : Flux optimizer
 - `ArrayType` : Option for using different type of array for storage (allow for GPU usage)
"""
function OnlineSVGP(#X::AbstractArray{T1},y::AbstractArray{T2},
            kernel::Kernel,
            likelihood::TLikelihood,inference::Inference,
            Z::InducingPoints=CircleKMeans(0.9,0.8),#,Sequential::Bool=false
            ;verbose::Integer=0,optimiser=ADAM(0.01),atfrequency::Integer=1,
            mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
            IndependentPriors::Bool=true,ArrayType::UnionAll=Vector,T₁=Float64) where {TLikelihood<:Likelihood}

            @assert inference isa AnalyticVI "The inference object should be of type `AnalyticVI`"
            @assert implemented(likelihood, inference) "The $likelihood is not compatible or implemented with the $inference"

            if isa(optimiser,Bool)
                optimiser = optimiser ? ADAM(0.01) : nothing
            end

            if typeof(mean) <: Real
                mean = ConstantMean(mean)
            elseif typeof(mean) <: AbstractVector{<:Real}
                mean = EmpiricalMean(mean)
            end

            nLatent = num_latent(likelihood)
            latentf = ntuple( _ -> _OSVGP{T₁}(0,0,Z,kernel,mean,optimiser),nLatent)
            inference = tuple_inference(inference,nLatent,0,0,0)
            inference.nIter = 1
            return OnlineSVGP{T₁,TLikelihood,typeof(inference),nLatent}(
                    Matrix{T₁}(undef,0,0), [],
                    0, 0, nLatent,
                    latentf, likelihood, inference,
                    verbose,atfrequency,false)
end

function Base.show(io::IO,model::OnlineSVGP{T,<:Likelihood,<:Inference}) where T
    print(io,"Online Variational Gaussian Process with a $(model.likelihood) infered by $(model.inference) ")
end

@traitimpl IsSparse{OnlineSVGP}

get_Z(model::OnlineSVGP) = get_Z.(model.f)
get_Z(model::OnlineSVGP, i::Int) = get_Z(model.f[i])
get_Zₐ(model::OnlineSVGP) = getproperty.(model.f,:Zₐ)
objective(model::OnlineSVGP) = ELBO(model)
