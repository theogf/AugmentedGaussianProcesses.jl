""" Class for variational Gaussian Processes models (non-sparse)"""
mutable struct VGP{L<:Likelihood,I<:Inference,T<:Real,V<:AbstractArray{T}} <: GP{L,I,T,V}
    X::V #Feature vectors
    y::LatentArray{V} #Output (-1,1 for classification, real for regression, matrix for multiclass)
    nSample::Int64 # Number of data points
    nDim::Int64 # Number of covariates per data point
    nFeature::Int64 # Number of features of the GP (equal to number of points)
    nLatent::Int64 # Number pf latent GPs
    IndependentPriors::Bool # Use of separate priors for each latent GP
    nPrior::Int64 # Equal to 1 or nLatent given IndependentPriors
    μ::LatentArray{V}
    Σ::LatentArray{Symmetric{T,Matrix{T}}}
    η₁::LatentArray{V}
    η₂::LatentArray{Symmetric{T,Matrix{T}}}
    Knn::LatentArray{Symmetric{T,Matrix{T}}}
    invKnn::LatentArray{Symmetric{T,Matrix{T}}}
    kernel::LatentArray{Kernel}
    likelihood::Likelihood{T}
    inference::Inference{T}
    verbose::Int64 #Level of printing information
    Autotuning::Bool
    atfrequency::Int64
    Trained::Bool
end

"""Create a variational Gaussian Process model
Argument list :

**Mandatory arguments**
 - `X` : input features, should be a matrix N×D where N is the number of observation and D the number of dimension
 - `y` : input labels, can be either a vector of labels for multiclass and single output or a matrix for multi-outputs (note that only one likelihood can be applied)
 - `kernel` : covariance function, can be either a single kernel or a collection of kernels for multiclass and multi-outputs models
 - `likelihood` : likelihood of the model, currently implemented : Gaussian, Bernoulli (with logistic link), Multiclass (softmax or logistic-softmax) see [`Likelihood`](@ref)
 - `inference` : inference for the model, can be analytic, numerical or by sampling, check the model documentation to know what is available for your likelihood see [`Inference`](@ref)
**Optional arguments**
 - `verbose` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
 - `Autotuning` : Flag for optimizing hyperparameters
 - `atfrequency` : Choose how many variational parameters iterations are between hyperparameters optimization
 - `IndependentPriors` : Flag for setting independent or shared parameters among latent GPs
 - `ArrayType` : Option for using different type of array for storage (allow for GPU usage)
"""
function VGP(X::AbstractArray{T1,N1},y::AbstractArray{T2,N2},kernel::Union{Kernel,AbstractVector{<:Kernel}},
            likelihood::LikelihoodType,inference::InferenceType;
            verbose::Integer=0,Autotuning::Bool=true,atfrequency::Integer=1,
            IndependentPriors::Bool=true,ArrayType::UnionAll=Array) where {T1<:Real,T2,N1,N2,LikelihoodType<:Likelihood,InferenceType<:Inference}

            X,y,likelihood = check_data!(X,y,likelihood)
            @assert check_implementation(likelihood,inference) "The $likelihood is not compatible or implemented with the $inference"

            nLatent = length(y);
            nPrior = IndependentPriors ? nLatent : 1
            nFeature = nSample = size(X,1); nDim = size(X,2);
            kernel = ArrayType([deepcopy(kernel) for _ in 1:nPrior])

            μ = LatentArray([zeros(T1,nFeature) for _ in 1:nLatent]); η₁ = copy(μ)
            Σ = LatentArray([Symmetric(ArrayType(Diagonal(ones(T1,nFeature)))) for _ in 1:nLatent])
            η₂ = inv.(Σ)*(-0.5);
            Knn = LatentArray([copy(Σ[1]) for _ in 1:nPrior]; invKnn = copy(Knn))

            likelihood = init_likelihood(likelihood,nLatent,nSample)
            inference = init_inference(inference,nLatent,nSample,nSample,nSample)

            VGP{LikelihoodType,InferenceType,T1,ArrayType{T1}}(X,y,
                    nFeature, nDim, nFeature, nLatent,
                    IndependentPriors,nPrior,μ,Σ,η₁,η₂,
                    Knn,invKnn,kernel,likelihood,inference,
                    verbose,Autotuning,atfrequency,false)
end

function Base.show(io::IO,model::VGP{<:Likelihood,<:Inference,T}) where T
    print(io,"Variational Gaussian Process with a $(model.likelihood) infered by $(model.inference) ")
end
