"""
Class for variational Gaussian Processes models (non-sparse)

```julia
VGP(X::AbstractArray{T1,N1},y::AbstractVector,kernel::Union{Kernel,AbstractVector{<:Kernel}},
    likelihood::LikelihoodType,inference::InferenceType;
    verbose::Int=0,optimizer::Union{Bool,Optimizer,Nothing}=Adam(α=0.01),atfrequency::Integer=1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
    IndependentPriors::Bool=true,ArrayType::UnionAll=Vector)
```

Argument list :

**Mandatory arguments**

 - `X` : input features, should be a matrix N×D where N is the number of observation and D the number of dimension
 - `y` : input labels, can be either a vector of labels for multiclass and single output or a matrix for multi-outputs (note that only one likelihood can be applied)
 - `kernel` : covariance function, can be either a single kernel or a collection of kernels for multiclass and multi-outputs models
 - `likelihood` : likelihood of the model, currently implemented : Gaussian, Bernoulli (with logistic link), Multiclass (softmax or logistic-softmax) see [`Likelihood Types`](@ref likelihood_user)
 - `inference` : inference for the model, can be analytic, numerical or by sampling, check the model documentation to know what is available for your likelihood see the [`Compatibility Table`](@ref compat_table)

**Keyword arguments**

 - `verbose` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
- `optimizer` : Optimizer for kernel hyperparameters (to be selected from [GradDescent.jl](https://github.com/jacobcvt12/GradDescent.jl))
- `atfrequency` : Choose how many variational parameters iterations are between hyperparameters optimization
- `mean` : PriorMean object, check the documentation on it [`MeanPrior`](@ref meanprior)
 - `IndependentPriors` : Flag for setting independent or shared parameters among latent GPs
 - `ArrayType` : Option for using different type of array for storage (allow for GPU usage)
"""
mutable struct VGP{T<:Real,TLikelihood<:Likelihood{T},TInference<:Inference{T},TGP<:Abstract_GP{T},N} <: AbstractGP{T,TLikelihood,TInference,TGP,N}
    X::Matrix{T} #Feature vectors
    y::Vector #Output (-1,1 for classification, real for regression, matrix for multiclass)
    nSamples::Int64 # Number of data points
    nDim::Int64 # Number of covariates per data point
    nFeatures::Int64 # Number of features of the GP (equal to number of points)
    nLatent::Int64 # Number pf latent GPs
    f::NTuple{N,TGP} # Vector of latent GPs
    likelihood::TLikelihood
    inference::TInference
    verbose::Int64 #Level of printing information
    atfrequency::Int64
    Trained::Bool
end


function VGP(X::AbstractArray{T},y::AbstractVector,kernel::Kernel,
            likelihood::TLikelihood,inference::TInference;
            verbose::Int=0,optimizer::Union{Bool,Optimizer,Nothing}=Adam(α=0.01),atfrequency::Integer=1,
            mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(), variance::Real = 1.0,
            ArrayType::UnionAll=Vector) where {T<:Real,TLikelihood<:Likelihood,TInference<:Inference}

            X, y, nLatent, likelihood = check_data!(X, y, likelihood)
            @assert check_implementation(:VGP, likelihood, inference) "The $likelihood is not compatible or implemented with the $inference"
            nFeatures = nSamples = size(X,1); nDim = size(X,2);

            if isa(optimizer,Bool)
                optimizer = optimizer ? Adam(α=0.01) : nothing
            end

            if typeof(mean) <: Real
                mean = ConstantMean(mean)
            elseif typeof(mean) <: AbstractVector{<:Real}
                mean = EmpiricalMean(mean)
            end

            latentf = ntuple(_->_VGP{T}(nFeatures,kernel,mean,variance,optimizer),nLatent)

            likelihood = init_likelihood(likelihood,inference,nLatent,nSamples,nFeatures)
            inference = tuple_inference(inference,nLatent,nSamples,nSamples,nSamples)
            inference.xview = view(X,:,:)
            inference.yview = view(y,:)
            inference.MBIndices = collect(1:nSamples)
            VGP{T,TLikelihood,typeof(inference),_VGP{T},nLatent}(X,y,
                    nFeatures, nDim, nFeatures, nLatent,
                    latentf,likelihood,inference,
                    verbose,atfrequency,false)
end

function Base.show(io::IO,model::VGP{T,<:Likelihood,<:Inference}) where {T}
    print(io,"Variational Gaussian Process with a $(model.likelihood) infered by $(model.inference) ")
end

const VGP1 = VGP{<:Real,<:Likelihood,<:Inference,<:Abstract_GP,1}

get_y(model::VGP) = model.inference.yview
get_X(model::VGP) = [model.inference.xview]
