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
mutable struct VStP{T<:Real,TLikelihood<:Likelihood{T},TInference<:Inference{T},V<:AbstractVector{T}} <: AbstractGP{T,TLikelihood,TInference,V}
    X::Matrix{T} #Feature vectors
    y::LatentArray #Output (-1,1 for classification, real for regression, matrix for multiclass)
    ν::T # Number of degrees of freedom
    l²::LatentArray{T} # Expectation of ||L^{-1}(f-μ⁰)||₂²
    χ::LatentArray{T} # Expectation of σ
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
    invL::LatentArray{LowerTriangular{T,Matrix{T}}}
    invKnn::LatentArray{Symmetric{T,Matrix{T}}}
    kernel::LatentArray{Kernel{T}}
    likelihood::TLikelihood
    inference::TInference
    verbose::Int64 #Level of printing information
    optimizer::Union{Optimizer,Nothing}
    atfrequency::Int64
    Trained::Bool
end


function VStP(X::AbstractArray{T₁,N₁},y::AbstractArray{T₂,N₂},kernel::Union{Kernel,AbstractVector{<:Kernel}},
            likelihood::TLikelihood,inference::TInference,ν::T₃;
            verbose::Int=0,optimizer::Union{Bool,Optimizer,Nothing}=Adam(α=0.01),atfrequency::Integer=1,
            mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
            IndependentPriors::Bool=true,ArrayType::UnionAll=Vector) where {T₁<:Real,T₂,T₃<:Real,N₁,N₂,TLikelihood<:Likelihood,TInference<:Inference}

            X,y,nLatent,likelihood = check_data!(X,y,likelihood)
            @assert check_implementation(:VStP,likelihood,inference) "The $likelihood is not compatible or implemented with the $inference"
            @assert ν > 1 "ν should be bigger than 1"
            nPrior = IndependentPriors ? nLatent : 1
            nFeatures = nSample = size(X,1); nDim = size(X,2);
            if isa(optimizer,Bool)
                optimizer = optimizer ? Adam(α=0.01) : nothing
            end
            if !isnothing(optimizer)
                setoptimizer!(kernel,optimizer)
            end
            kernel = ArrayType([deepcopy(kernel) for _ in 1:nPrior])

            μ = LatentArray([zeros(T1,nFeatures) for _ in 1:nLatent]); η₁ = deepcopy(μ)
            Σ = LatentArray([Symmetric(Matrix(Diagonal(one(T1)*I,nFeatures))) for _ in 1:nLatent]);
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
            L = LatentArray([LowerTriangular(rand(T1,nSample,nSample)) for _ in 1:nPrior])
            invKnn = copy(Knn)

            l² = LatentArray(rand(T1,nLatent))
            χ = LatentArray(rand(T1,nLatent))

            likelihood = init_likelihood(likelihood,inference,nLatent,nSample,nFeatures)
            inference = init_inference(inference,nLatent,nSample,nSample,nSample)
            inference.x = view(X,:,:)
            inference.y = view.(y,:)
            VStP{T1,TLikelihood,TInference,ArrayType{T1}}(X,y,ν,
                    l² ,χ, nFeatures, nDim, nFeatures, nLatent,
                    IndependentPriors,nPrior,μ,Σ,η₁,η₂,
                    μ₀,Knn,L,invKnn,kernel,likelihood,inference,
                    verbose,optimizer,atfrequency,false)
end

function Base.show(io::IO,model::VStP{T,<:Likelihood,<:Inference}) where {T}
    print(io,"Variational Student-T Process with a $(model.likelihood) infered by $(model.inference) ")
end


@inline invK(model::VStP) = inv.(model.χ).*model.invKnn
@inline invK(model::VStP,i::Integer) = inv(model.χ[i])*model.invKnn[i]

function local_prior_updates!(model::VStP)
    model.l² .= broadcast((ν,μ,Σ,μ₀,invK)->0.5*(ν+model.nSample+dot(μ-μ₀,invK*(μ-μ₀))+opt_trace(invK,Σ)),model.ν,model.μ,model.Σ,model.μ₀,model.invKnn)
    model.χ .= (model.ν.+model.nSample)./(model.ν.+model.l²)
end
