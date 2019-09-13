"""
Class for Gaussian Processes models

```julia
GP(X::AbstractArray{T₁,N₁}, y::AbstractArray{T₂,N₂}, kernel::Union{Kernel,AbstractVector{<:Kernel}};
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
mutable struct GP{T<:Real,TLikelihood<:Likelihood{T},TInference<:Inference{T},V<:AbstractVector{T}} <: AbstractGP{T,TLikelihood,TInference,V}
    X::Matrix{T} #Feature vectors
    y::LatentArray #Output (-1,1 for classification, real for regression, matrix for multiclass)
    nSample::Int64 # Number of data points
    nDim::Int64 # Number of covariates per data point
    nFeatures::Int64 # Number of features of the GP (equal to number of points)
    nLatent::Int64 # Number pf latent GPs
    IndependentPriors::Bool # Use of separate priors for each latent GP
    nPrior::Int64 # Equal to 1 or nLatent given IndependentPriors
    μ₀::LatentArray{PriorMean{T}}
    Knn::LatentArray{Symmetric{T,Matrix{T}}}
    invKnn::LatentArray{Symmetric{T,Matrix{T}}}
    kernel::LatentArray{Kernel{T}}
    likelihood::TLikelihood
    inference::TInference
    verbose::Int64 #Level of printing information
    optimizer::Union{Optimizer,Nothing}
    atfrequency::Int64
    opt_noise::Bool
    Trained::Bool
end


function GP(X::AbstractArray{T₁,N₁}, y::AbstractArray{T₁,N₂}, kernel::Union{Kernel,AbstractVector{<:Kernel}};
                noise::Real=1e-5, opt_noise::Bool=true, verbose::Int=0,
                optimizer::Union{Optimizer,Nothing,Bool}=Adam(α=0.01),atfrequency::Int=1,
                mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
                IndependentPriors::Bool=true,ArrayType::UnionAll=Vector) where {T₁<:Real,T₂,N₁,N₂}
            likelihood = GaussianLikelihood(noise)
            inference = Analytic()
            X,y,nLatent,likelihood = check_data!(X,y,likelihood)

            nPrior = IndependentPriors ? nLatent : 1
            nFeatures = nSample = size(X,1); nDim = size(X,2);
            if isa(optimizer,Bool)
                optimizer = optimizer ? Adam(α=0.01) : nothing
            end
            if !isnothing(optimizer)
                setoptimizer!(kernel,optimizer)
            end
            kernel = ArrayType([deepcopy(kernel) for _ in 1:nPrior])

            Knn = LatentArray([Symmetric(Matrix{T1}(I,nFeatures,nFeatures)) for _ in 1:nPrior]);
            invKnn = copy(Knn)
            μ₀ = []
            if typeof(mean) <: Real
                μ₀ = [ConstantMean(mean) for _ in 1:nPrior]
            elseif typeof(mean) <: AbstractVector{<:Real}
                μ₀ = [EmpiricalMean(mean) for _ in 1:nPrior]
            else
                μ₀ = [mean for _ in 1:nPrior]
            end
            likelihood = init_likelihood(likelihood,inference,nLatent,nSample,nFeatures)
            inference = init_inference(inference,nLatent,nSample,nSample,nSample)
            inference.x = view(X,:,:)
            inference.y = view.(y,:)
            model = GP{T1,GaussianLikelihood{T1},Analytic{T1},ArrayType{T1}}(X,y,
                    nFeatures, nDim, nFeatures, nLatent,
                    IndependentPriors,nPrior,
                    μ₀,Knn,invKnn,kernel,likelihood,inference,
                    verbose,optimizer,atfrequency,opt_noise,false)
            computeMatrices!(model)
            model.Trained = true
            return model
end

function Base.show(io::IO,model::GP{T,<:Likelihood,<:Inference}) where {T}
    print(io,"Gaussian Process with a $(model.likelihood) infered by $(model.inference) ")
end

@inline invK(model::GP) = model.invKnn
@inline invK(model::GP,i::Integer) = model.invKnn[i]
