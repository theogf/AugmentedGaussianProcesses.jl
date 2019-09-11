"""
Class for variational Gaussian Processes models (non-sparse)

```julia
VGP(X::AbstractArray{T₁,N₁},y::AbstractArray{T₂,N₂},kernel::Union{Kernel,AbstractVector{<:Kernel}},
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
- `optimizer` : Optimizer for kernel hyperparameters (to be selected from [GradDescent.jl](https://github.com/jacobcvt12/GradDescent.jl)) or set it to `false` to keep hyperparameters fixed
- `atfrequency` : Choose how many variational parameters iterations are between hyperparameters optimization
- `mean` : PriorMean object, check the documentation on it [`MeanPrior`](@ref meanprior)
 - `IndependentPriors` : Flag for setting independent or shared parameters among latent GPs
 - `ArrayType` : Option for using different type of array for storage (allow for GPU usage)
"""
mutable struct VGP{T<:Real,TLikelihood<:Likelihood{T},TInference<:Inference{T},V<:AbstractVector{T}} <: AbstractGP{T,TLikelihood,TInference,V}
    X::Matrix{T} #Feature vectors
    y::LatentArray #Output (-1,1 for classification, real for regression, matrix for multiclass)
    nSample::Int64 # Number of data points
    nDim::Int64 # Number of covariates per data point
    nFeatures::Int64 # Number of features of the GP (equal to number of points)
    nLatent::Int64 # Number pf latent GPs
    IndependentPriors::Bool # Use of separate priors for each latent GP
    nPrior::Int64 # Equal to 1 or nLatent given IndependentPriors
    μ::LatentArray{V}
    Σ::LatentArray{Symmetric{T,Matrix{T}}}
    η₁::LatentArray{V}
    η₂::LatentArray{Symmetric{T,Matrix{T}}}
    μ₀::LatentArray{PriorMean{T}}
    Knn::LatentArray{Symmetric{T,Matrix{T}}}
    invKnn::LatentArray{Symmetric{T,Matrix{T}}}
    kernel::LatentArray{Kernel{T}}
    likelihood::TLikelihood
    inference::TInference
    verbose::Int64 #Level of printing information
    optimizer::Union{Optimizer,Nothing}
    atfrequency::Int64
    Trained::Bool
end


function VGP(X::AbstractArray{T₁,N₁},y::AbstractArray{T₂,N₂},kernel::Union{Kernel,AbstractVector{<:Kernel}},
            likelihood::TLikelihood,inference::TInference;
            verbose::Int=0,optimizer::Union{Bool,Optimizer,Nothing}=Adam(α=0.01),atfrequency::Integer=1,
            mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
            IndependentPriors::Bool=true,ArrayType::UnionAll=Vector) where {T₁<:Real,T₂,N₁,N₂,TLikelihood<:Likelihood,TInference<:Inference}

            X,y,nLatent,likelihood = check_data!(X,y,likelihood)
            @assert check_implementation(:VGP,likelihood,inference) "The $likelihood is not compatible or implemented with the $inference"

            nPrior = IndependentPriors ? nLatent : 1
            nFeatures = nSample = size(X,1); nDim = size(X,2);
            if isa(optimizer,Bool)
                optimizer = optimizer ? Adam(α=0.01) : nothing
            end
            if !isnothing(optimizer) && isa(inference,GibbsSampling)
                @warn "Hyperparameter optimization is not available with Gibbs Sampling, disabling it"
                optimizer = nothing
            end
            if !isnothing(optimizer)
                setoptimizer!(kernel,optimizer)
            end

            kernel = ArrayType([deepcopy(kernel) for _ in 1:nPrior])

            μ = LatentArray([zeros(T₁,nFeatures) for _ in 1:nLatent]); η₁ = deepcopy(μ)
            Σ = LatentArray([Symmetric(Matrix(Diagonal(one(T₁)*I,nFeatures))) for _ in 1:nLatent]);
            η₂ = -0.5*inv.(Σ);
            μ₀ = []
            if typeof(mean) <: Real
                μ₀ = [ConstantMean(mean) for _ in 1:nPrior]
            elseif typeof(mean) <: AbstractVector{<:Real}
                μ₀ = [EmpiricalMean(mean) for _ in 1:nPrior]
            else
                μ₀ = [mean for _ in 1:nPrior]
            end
            Knn = LatentArray([deepcopy(Σ[1]) for _ in 1:nPrior]);
            invKnn = copy(Knn)

            likelihood = init_likelihood(likelihood,inference,nLatent,nSample,nFeatures)
            inference = init_inference(inference,nLatent,nSample,nSample,nSample)
            inference.x = view(X,:,:)
            inference.y = view.(y,:)
            VGP{T₁,TLikelihood,TInference,ArrayType{T₁}}(X,y,
                    nFeatures, nDim, nFeatures, nLatent,
                    IndependentPriors,nPrior,μ,Σ,η₁,η₂,
                    μ₀,Knn,invKnn,kernel,likelihood,inference,
                    verbose,optimizer,atfrequency,false)
end

function Base.show(io::IO,model::VGP{T,<:Likelihood,<:Inference}) where {T}
    print(io,"Variational Gaussian Process with a $(model.likelihood) infered by $(model.inference) ")
end

@inline invK(model::VGP) = model.invKnn
@inline invK(model::VGP,i::Integer) = model.invKnn[i]
