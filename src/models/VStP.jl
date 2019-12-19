"""
Class for variational Student-T Processes models (non-sparse)

```julia
VStP(X::AbstractArray{T₁,N₁},y::AbstractArray{T₂,N₂},kernel::Union{Kernel,AbstractVector{<:Kernel}},
    likelihood::LikelihoodType,inference::InferenceType,ν::T₃;
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
 - `ν` : Number of degrees of freedom

**Keyword arguments**

 - `verbose` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
- `optimizer` : Optimizer for kernel hyperparameters (to be selected from [GradDescent.jl](https://github.com/jacobcvt12/GradDescent.jl)) or set it to `false` to keep hyperparameters fixed
- `atfrequency` : Choose how many variational parameters iterations are between hyperparameters optimization
- `mean` : PriorMean object, check the documentation on it [`MeanPrior`](@ref meanprior)
 - `IndependentPriors` : Flag for setting independent or shared parameters among latent GPs
 - `ArrayType` : Option for using different type of array for storage (allow for GPU usage)
"""
mutable struct VStP{T<:Real,TLikelihood<:Likelihood{T},TInference<:Inference{T},N} <: AbstractGP{T,TLikelihood,TInference,N}
    X::Matrix{T} #Feature vectors
    y::LatentArray #Output (-1,1 for classification, real for regression, matrix for multiclass)
    ν::T # Number of degrees of freedom
    nSample::Int64 # Number of data points
    nDim::Int64 # Number of covariates per data point
    nFeatures::Int64 # Number of features of the GP (equal to number of points)
    nLatent::Int64 # Number pf latent GPs
    f::NTuple{N,_VStP}
    likelihood::TLikelihood
    inference::TInference
    verbose::Int64 #Level of printing information
    atfrequency::Int64
    Trained::Bool
end


function VStP(X::AbstractArray{T},y::AbstractVector,kernel::Kernel,
            likelihood::TLikelihood,inference::TInference,ν::Real;
            verbose::Int=0,optimizer=ADAM(0.01),atfrequency::Integer=1,
            mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(), variance::Real = 1.0,
            ArrayType::UnionAll=Vector) where {T<:Real,TLikelihood<:Likelihood,TInference<:Inference}

            X,y,nLatent,likelihood = check_data!(X,y,likelihood)
            @assert check_implementation(:VStP,likelihood,inference) "The $likelihood is not compatible or implemented with the $inference"
            @assert ν > 1 "ν should be bigger than 1"
            nFeatures = nSample = size(X,1); nDim = size(X,2);

            if isa(optimizer,Bool)
                optimizer = optimizer ? ADAM(0.01) : nothing
            end

            if typeof(mean) <: Real
                mean = ConstantMean(mean)
            elseif typeof(mean) <: AbstractVector{<:Real}
                mean = EmpiricalMean(mean)
            end

            latentf = ntuple(_->_VStP{T}(ν,nFeatures,kernel,mean,variance,optimizer),nLatent)

            likelihood = init_likelihood(likelihood,inference,nLatent,nSamples,nFeatures)
            inference = tuple_inference(inference,nLatent,nSamples,nSamples,nSamples)
            inference.x = view(X,:,:)
            inference.y = view_y(likelihood,y,1:nSamples)
            inference.MBIndices = collect(1:nSamples)
            VStP{T1,TLikelihood,typeof(inference),nLatent}(
                    X,y,ν, nFeatures, nDim, nFeatures, nLatent,
                    latentf, likelihood, inference,
                    verbose, atfrequency, false)
end

function Base.show(io::IO,model::VStP)
    print(io,"Variational Student-T Process with a $(model.likelihood) infered by $(model.inference) ")
end


@inline invK(model::VStP) = inv.(model.χ).*model.invKnn
@inline invK(model::VStP,i::Integer) = inv(model.χ[i])*model.invKnn[i]

function local_prior_updates!(model::VStP)
    model.l² .= broadcast((ν,μ,Σ,μ₀,invK)->0.5*(ν+model.nSample+dot(μ-μ₀,invK*(μ-μ₀))+opt_trace(invK,Σ)),model.ν,model.μ,model.Σ,model.μ₀,model.invKnn)
    model.χ .= (model.ν.+model.nSample)./(model.ν.+model.l²)
end

@traitimpl IsFull{VStP}
