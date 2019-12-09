"""
Class for Gaussian Processes models

```julia
GP(X::AbstractArray{T}, y::AbstractArray, kernel::Kernel;
    noise::Real=1e-5, opt_noise::Bool=true, verbose::Int=0,
    optimizer::Bool=Adam(α=0.01),atfrequency::Int=1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
    IndependentPriors::Bool=true,ArrayType::UnionAll=Vector)
```

Argument list :

**Mandatory arguments**

 - `X` : input features, should be a matrix N×D where N is the number of observation and D the number of dimension
 - `y` : input labels, can be either a vector of labels for multiclass and single output or a matrix for multi-outputs (note that only one likelihood can be applied)
 - `kernel` : covariance function, can be either a single kernel or a collection of kernels for multiclass and multi-outputs models

**Keyword arguments**
 - `noise` : Initial noise of the model
 - `opt_noise` : Flag for optimizing the noise σ=Σ(y-f)^2/N
 - `mean` : Option for putting a prior mean
 - `verbose` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
 - `optimizer` : Optimizer for kernel hyperparameters (to be selected from [GradDescent.jl](https://github.com/jacobcvt12/GradDescent.jl)) or set it to `false` to keep hyperparameters fixed
 - `IndependentPriors` : Flag for setting independent or shared parameters among latent GPs
 - `atfrequency` : Choose how many variational parameters iterations are between hyperparameters optimization
 - `mean` : PriorMean object, check the documentation on it [`MeanPrior`](@ref meanprior)
 - `ArrayType` : Option for using different type of array for storage (allow for GPU usage)
"""
mutable struct GP{T<:Real,TLikelihood<:Likelihood{T},TInference<:Inference{T},N} <: AbstractGP{T,TLikelihood,TInference,N}
    X::Matrix{T} #Feature vectors
    y::Vector #Output (-1,1 for classification, real for regression, matrix for multiclass)
    nSamples::Int64 # Number of data points
    nDim::Int64 # Number of covariates per data point
    nFeatures::Int64 # Number of features of the GP (equal to number of points)
    nLatent::Int64 # Number pf latent GPs
    f::NTuple{N,_GP} # Vector of latent GPs
    likelihood::TLikelihood
    inference::TInference
    verbose::Int64 #Level of printing information
    atfrequency::Int64
    Trained::Bool
end


function GP(X::AbstractArray{T}, y::AbstractArray,kernel::Kernel;
                noise::Real=1e-5, opt_noise::Bool=true, verbose::Int=0,
                optimizer::Union{Optimizer,Nothing,Bool}=Adam(α=0.01),atfrequency::Int=1,
                mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(), variance::Real = 1.0,
                ArrayType::UnionAll=Vector) where {T<:Real}
            likelihood = GaussianLikelihood(noise,opt_noise=opt_noise)
            inference = Analytic()
            X,y,nLatent,likelihood = check_data!(X,y,likelihood)

            nFeatures = nSamples = size(X,1); nDim = size(X,2);
            if isa(optimizer,Bool)
                optimizer = optimizer ? Adam(α=0.01) : nothing
            end

            latentf = ntuple(_->_GP{T}(nFeatures,kernel,mean,variance,optimizer),nLatent)

            likelihood = init_likelihood(likelihood,inference,nLatent,nSamples,nFeatures)
            inference = init_inference(inference,nLatent,nSamples,nSamples,nSamples)
            inference.xview = view(X,:,:)
            inference.yview = view_y(likelihood,y,1:nSamples)
            model = GP{T,GaussianLikelihood{T},typeof(inference),_GP{T},1}(X,y,nFeatures,
            nDim, nFeatures, nLatent,latentf,likelihood,inference,
            verbose,atfrequency,false)
            computeMatrices!(model)
            model.Trained = true
            return model
end

function Base.show(io::IO,model::GP{T,<:Likelihood,<:Inference}) where {T}
    print(io,"Gaussian Process with a $(model.likelihood) infered by $(model.inference) ")
end

get_y(model::GP) = model.inference.yview
get_Z(model::GP) = [model.inference.xview]

@traitimpl IsFull{GP}
