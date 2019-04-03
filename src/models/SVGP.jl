""" Class for sparse variational Gaussian Processes """
mutable struct SVGP{L<:Likelihood,I<:Inference,T<:Real,V<:AbstractVector{T}} <: SparseGP{L,I,T,V}
    X::Matrix{T} #Feature vectors
    y::LatentArray #Output (-1,1 for classification, real for regression, matrix for multiclass)
    nSample::Int64 # Number of data points
    nDim::Int64 # Number of covariates per data point
    nFeature::Int64 # Number of features of the GP (equal to number of points)
    nLatent::Int64 # Number pf latent GPs
    IndependentPriors::Bool # Use of separate priors for each latent GP
    nPrior::Int64 # Equal to 1 or nLatent given IndependentPriors
    Z::LatentArray{Matrix{T}} #Inducing points locations
    μ::LatentArray{V}
    Σ::LatentArray{Symmetric{T,Matrix{T}}}
    η₁::LatentArray{V}
    η₂::LatentArray{Symmetric{T,Matrix{T}}}
    Kmm::LatentArray{Symmetric{T,Matrix{T}}}
    invKmm::LatentArray{Symmetric{T,Matrix{T}}}
    Knm::LatentArray{Matrix{T}}
    κ::LatentArray{Matrix{T}}
    K̃::LatentArray{V}
    kernel::LatentArray{Kernel{T}}
    likelihood::Likelihood{T}
    inference::Inference{T}
    verbose::Int64
    Autotuning::Bool
    atfrequency::Int64
    OptimizeInducingPoints::Bool
    Trained::Bool
end

"""Create a sparse variational Gaussian Process model
Argument list :

**Mandatory arguments**
 - `X` : input features, should be a matrix N×D where N is the number of observation and D the number of dimension
 - `y` : input labels, can be either a vector of labels for multiclass and single output or a matrix for multi-outputs (note that only one likelihood can be applied)
 - `kernel` : covariance function, can be either a single kernel or a collection of kernels for multiclass and multi-outputs models
 - `likelihood` : likelihood of the model, currently implemented : Gaussian, Bernoulli (with logistic link), Multiclass (softmax or logistic-softmax) see [`Likelihood`](@ref)
 - `inference` : inference for the model, can be analytic, numerical or by sampling, check the model documentation to know what is available for your likelihood see [`Inference`](@ref)
 - `nInducingPoints` : number of inducing points
**Optional arguments**
 - `verbose` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
 - `Autotuning` : Flag for optimizing hyperparameters
 - `atfrequency` : Choose how many variational parameters iterations are between hyperparameters optimization
 - `IndependentPriors` : Flag for setting independent or shared parameters among latent GPs
 - `OptimizeInducingPoints` : Flag for optimizing the inducing points locations
 - `ArrayType` : Option for using different type of array for storage (allow for GPU usage)
"""
function SVGP(X::AbstractArray{T1},y::AbstractArray{T2},kernel::Kernel,
            likelihood::LikelihoodType,inference::InferenceType,
            nInducingPoints::Integer
            ;verbose::Integer=0,Autotuning::Bool=true,atfrequency::Integer=1,
            IndependentPriors::Bool=true, OptimizeInducingPoints::Bool=false,ArrayType::UnionAll=Vector) where {T1<:Real,T2,LikelihoodType<:Likelihood,InferenceType<:Inference}

            X,y,nLatent,likelihood = check_data!(X,y,likelihood)
            @assert check_implementation(:SVGP,likelihood,inference) "The $likelihood is not compatible or implemented with the $inference"

            nPrior = IndependentPriors ? nLatent : 1
            nSample = size(X,1); nDim = size(X,2);
            kernel = [deepcopy(kernel) for _ in 1:nPrior]


            @assert nInducingPoints > 0 && nInducingPoints < nSample "The number of inducing points is incorrect (negative or bigger than number of samples)"
            Z = KMeansInducingPoints(X,nInducingPoints,nMarkov=10); Z=[deepcopy(Z) for _ in 1:nPrior]
            nFeature = nInducingPoints


            μ = LatentArray([zeros(T1,nFeature) for _ in 1:nLatent]); η₁ = deepcopy(μ);
            Σ = LatentArray([Symmetric(Matrix(Diagonal(one(T1)*I,nFeature))) for _ in 1:nLatent]);
            η₂ = -0.5*inv.(Σ);
            κ = LatentArray([zeros(T1,inference.Stochastic ? inference.nSamplesUsed : nSample, nFeature) for _ in 1:nPrior])
            Knm = deepcopy(κ)
            K̃ = LatentArray([zeros(T1,inference.Stochastic ? inference.nSamplesUsed : nSample) for _ in 1:nPrior])
            Kmm = LatentArray([similar(Σ[1]) for _ in 1:nPrior]); invKmm = similar.(Kmm)
            nSamplesUsed = nSample
            if inference.Stochastic
                @assert inference.nSamplesUsed > 0 && inference.nSamplesUsed < nSample "The size of mini-batch is incorrect (negative or bigger than number of samples), please set nMinibatch correctly in the inference object"
                nSamplesUsed = inference.nSamplesUsed
            end

            likelihood = init_likelihood(likelihood,inference,nLatent,nSamplesUsed)
            inference = init_inference(inference,nLatent,nFeature,nSample,nSamplesUsed)
            model = SVGP{LikelihoodType,InferenceType,T1,ArrayType{T1}}(X,y,
                    nSample, nDim, nFeature, nLatent,
                    IndependentPriors,nPrior,
                    Z,μ,Σ,η₁,η₂,
                    Kmm,invKmm,Knm,κ,K̃,
                    kernel,likelihood,inference,
                    verbose,Autotuning,atfrequency,OptimizeInducingPoints,false)
            if isa(inference.optimizer_η₁[1],ALRSVI)
                init!(model.inference,model)
            end
            return model
end

function Base.show(io::IO,model::SVGP{<:Likelihood,<:Inference,T}) where T
    print(io,"Sparse Variational Gaussian Process with a $(model.likelihood) infered by $(model.inference) ")
end
