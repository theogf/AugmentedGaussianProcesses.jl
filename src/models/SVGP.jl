"""
Class for sparse variational Gaussian Processes

```julia
SVGP(X::AbstractArray{T1},y::AbstractArray{T2},kernel::Union{Kernel,AbstractVector{<:Kernel}},
    likelihood::LikelihoodType,inference::InferenceType, nInducingPoints::Int;
    verbose::Int=0,optimizer::Union{Optimizer,Nothing,Bool}=Adam(α=0.01),atfrequency::Int=1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
    IndependentPriors::Bool=true,Zoptimizer::Union{Optimizer,Nothing,Bool}=false,
    ArrayType::UnionAll=Vector)
```

Argument list :

**Mandatory arguments**
 - `X` : input features, should be a matrix N×D where N is the number of observation and D the number of dimension
 - `y` : input labels, can be either a vector of labels for multiclass and single output or a matrix for multi-outputs (note that only one likelihood can be applied)
 - `kernel` : covariance function, can be either a single kernel or a collection of kernels for multiclass and multi-outputs models
 - `likelihood` : likelihood of the model, currently implemented : Gaussian, Student-T, Laplace, Bernoulli (with logistic link), Bayesian SVM, Multiclass (softmax or logistic-softmax) see [`Likelihood`](@ref likelihood_user)
 - `inference` : inference for the model, can be analytic, numerical or by sampling, check the model documentation to know what is available for your likelihood see the [`Compatibility table`](@ref compat_table)
 - `nInducingPoints` : number of inducing points
**Optional arguments**
 - `verbose` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
 - `optimizer` : Optimizer for kernel hyperparameters (to be selected from [GradDescent.jl](https://github.com/jacobcvt12/GradDescent.jl))
 - `atfrequency` : Choose how many variational parameters iterations are between hyperparameters optimization
 - `mean` : PriorMean object, check the documentation on it [`MeanPrior`](@ref meanprior)
 - `IndependentPriors` : Flag for setting independent or shared parameters among latent GPs
 - `optimizer` : Optimizer for inducing point locations (to be selected from [GradDescent.jl](https://github.com/jacobcvt12/GradDescent.jl))
 - `ArrayType` : Option for using different type of array for storage (allow for GPU usage)
"""
mutable struct SVGP{T<:Real,TLikelihood<:Likelihood{T},TInference<:Inference,TGP<:Abstract_GP{T},N} <: AbstractGP{T,TLikelihood,TInference,TGP,N}
    X::Matrix{T} #Feature vectors
    y::Vector #Output (-1,1 for classification, real for regression, matrix for multiclass)
    nSample::Int64 # Number of data points
    nDim::Int64 # Number of covariates per data point
    nFeatures::Int64 # Number of features of the GP (equal to number of points)
    nLatent::Int64 # Number pf latent GPs
    IndependentPriors::Bool # Use of separate priors for each latent GP
    nPrior::Int64 # Equal to 1 or nLatent given IndependentPriors
    f::NTuple{N,TGP}
    likelihood::TLikelihood
    inference::TInference
    verbose::Int64
    optimizer::Union{Optimizer,Nothing}
    atfrequency::Int64
    Trained::Bool
end

function SVGP(X::AbstractArray{T1},y::AbstractArray{T2},kernel::Union{Kernel,AbstractVector{<:Kernel}},
            likelihood::TLikelihood,inference::TInference, nInducingPoints::Int;
            verbose::Int=0,optimizer::Union{Optimizer,Nothing,Bool}=Adam(α=0.01),atfrequency::Int=1,
            mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(), variance::Real = 1.0,
            IndependentPriors::Bool=true,Zoptimizer::Union{Optimizer,Nothing,Bool}=false,
            ArrayType::UnionAll=Vector) where {T1<:Real,T2,TLikelihood<:Likelihood,TInference<:Inference}

            X,y,nLatent,likelihood = check_data!(X,y,likelihood)
            @assert check_implementation(:SVGP,likelihood,inference) "The $likelihood is not compatible or implemented with the $inference"

            nPrior = IndependentPriors ? nLatent : 1
            nSample = size(X,1); nDim = size(X,2);
            if isa(optimizer,Bool)
                optimizer = optimizer ? Adam(α=0.01) : nothing
            end
            # if !isnothing(optimizer)
                # setoptimizer!(kernel,optimizer)
            # end

            @assert nInducingPoints > 0 && nInducingPoints < nSample "The number of inducing points is incorrect (negative or bigger than number of samples)"
            Z = KMeansInducingPoints(X,nInducingPoints,nMarkov=10)
            Zoptimizer = !Zoptimizer ? nothing : Zoptimizer
            Z = InducingPoints(Z,Zoptimizer)

            nFeatures = nInducingPoints

            if typeof(mean) <: Real
                mean = ConstantMean(mean)
            elseif typeof(mean) <: AbstractVector{<:Real}
                mean = EmpiricalMean(mean)
            end

            nSamplesUsed = nSample
            if inference.Stochastic
                @assert inference.nSamplesUsed > 0 && inference.nSamplesUsed < nSample "The size of mini-batch is incorrect (negative or bigger than number of samples), please set nMinibatch correctly in the inference object"
                nSamplesUsed = inference.nSamplesUsed
            end

            latentf = ntuple(_->_SVGP{T1}(nFeatures,nSamplesUsed,Z,kernel,mean,variance),nLatent)

            likelihood = init_likelihood(likelihood,inference,nLatent,nSamplesUsed,nFeatures)
            inference = init_inference(inference,nLatent,nFeatures,nSample,nSamplesUsed)
            inference.x = view(X,1:nSample,:)
            inference.y = view(y,:)

            model = SVGP{T1,TLikelihood,TInference,_SVGP{T1},nLatent}(X,y,
                    nSample, nDim, nFeatures, nLatent,
                    IndependentPriors,nPrior,
                    latentf,likelihood,inference,
                    verbose,optimizer,atfrequency,false)
            if isa(inference.optimizer,ALRSVI)
                init!(model.inference,model)
            end
            return model
end

function Base.show(io::IO,model::SVGP{T,<:Likelihood,<:Inference}) where {T}
    print(io,"Sparse Variational Gaussian Process with a $(model.likelihood) infered by $(model.inference) ")
end

const SVGP1 = SVGP{<:Real,<:Likelihood,<:Inference,<:Abstract_GP,1}

get_y(model::SVGP) = model.inference.y
get_X(model::SVGP1) = model.f[1].Z.Z
