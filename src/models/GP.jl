"Class for variational Gaussian Processes models (non-sparse)"
mutable struct GP{L<:Likelihood,I<:Inference,T<:Real,V<:AbstractVector{T}} <: AbstractGP{L,I,T,V}
    X::Matrix{T} #Feature vectors
    y::LatentArray #Output (-1,1 for classification, real for regression, matrix for multiclass)
    nSample::Int64 # Number of data points
    nDim::Int64 # Number of covariates per data point
    nFeature::Int64 # Number of features of the GP (equal to number of points)
    nLatent::Int64 # Number pf latent GPs
    IndependentPriors::Bool # Use of separate priors for each latent GP
    nPrior::Int64 # Equal to 1 or nLatent given IndependentPriors
    Knn::LatentArray{Symmetric{T,Matrix{T}}}
    invKnn::LatentArray{Symmetric{T,Matrix{T}}}
    kernel::LatentArray{Kernel{T}}
    likelihood::Likelihood{T}
    inference::Inference{T}
    verbose::Int64 #Level of printing information
    Autotuning::Bool
    atfrequency::Int64
    Trained::Bool
end

"""Create a Gaussian Process model
Argument list :

**Mandatory arguments**
 - `X` : input features, should be a matrix N×D where N is the number of observation and D the number of dimension
 - `y` : input labels, can be either a vector of labels for multiclass and single output or a matrix for multi-outputs (note that only one likelihood can be applied)
 - `kernel` : covariance function, can be either a single kernel or a collection of kernels for multiclass and multi-outputs models
**Optional arguments**
 - `verbose` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
 - `Autotuning` : Flag for optimizing hyperparameters
 - `atfrequency` : Choose how many variational parameters iterations are between hyperparameters optimization
 - `IndependentPriors` : Flag for setting independent or shared parameters among latent GPs
 - `ArrayType` : Option for using different type of array for storage (allow for GPU usage)
"""
function GP(X::AbstractArray{T1,N1},y::AbstractArray{T2,N2},kernel::Union{Kernel,AbstractVector{<:Kernel}};  noise::Real=1e-5,
            verbose::Integer=0,Autotuning::Bool=true,atfrequency::Integer=1,
            IndependentPriors::Bool=true,ArrayType::UnionAll=Vector) where {T1<:Real,T2,N1,N2}
            likelihood = GaussianLikelihood(noise)
            inference = Analytic()
            X,y,nLatent,likelihood = check_data!(X,y,likelihood)

            nPrior = IndependentPriors ? nLatent : 1
            nFeature = nSample = size(X,1); nDim = size(X,2);
            kernel = ArrayType([deepcopy(kernel) for _ in 1:nPrior])

            Knn = LatentArray([Symmetric(Matrix{T1}(I,nFeature,nFeature)) for _ in 1:nPrior]);
            invKnn = copy(Knn)

            likelihood = init_likelihood(likelihood,inference,nLatent,nSample)
            inference = init_inference(inference,nLatent,nSample,nSample,nSample)

            model = GP{GaussianLikelihood{T1},Analytic{T1},T1,ArrayType{T1}}(X,y,
                    nFeature, nDim, nFeature, nLatent,
                    IndependentPriors,nPrior,
                    Knn,invKnn,kernel,likelihood,inference,
                    verbose,Autotuning,atfrequency,false)
            computeMatrices!(model)
            model.Trained = true
            return model
end

function Base.show(io::IO,model::GP{<:Likelihood,<:Inference,T}) where T
    print(io,"Gaussian Process with a $(model.likelihood) infered by $(model.inference) ")
end
