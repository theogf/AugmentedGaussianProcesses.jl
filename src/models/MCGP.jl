"""
Class for variational Gaussian Processes models (non-sparse)

```julia
MCGP(X::AbstractArray{T1,N1},y::AbstractArray{T2,N2},kernel::Union{Kernel,AbstractVector{<:Kernel}},
    likelihood::LikelihoodType,inference::InferenceType;
    verbose::Int=0,optimiser=ADAM(0.01),atfrequency::Integer=1,
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
- `optimiser` : Optimiser used for the kernel parameters. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `ADAM(0.001)`
- `atfrequency` : Choose how many variational parameters iterations are between hyperparameters optimization
- `mean` : PriorMean object, check the documentation on it [`MeanPrior`](@ref meanprior)
 - `IndependentPriors` : Flag for setting independent or shared parameters among latent GPs
 - `ArrayType` : Option for using different type of array for storage (allow for GPU usage)
"""
mutable struct MCGP{T<:Real,TLikelihood<:Likelihood{T},TInference<:Inference{T},N} <: AbstractGP{T,TLikelihood,TInference,N}
    X::Matrix{T} #Feature vectors
    y::LatentArray #Output (-1,1 for classification, real for regression, matrix for multiclass)
    nSamples::Int64 # Number of data points
    nDim::Int64 # Number of covariates per data point
    nFeatures::Int64 # Number of features of the GP (equal to number of points)
    nLatent::Int64 # Number pf latent GPs
    f::NTuple{N,_MCGP} # Vector of latent GPs
    likelihood::TLikelihood
    inference::TInference
    verbose::Int64 #Level of printing information
    atfrequency::Int64
    Trained::Bool
end


function MCGP(
    X::AbstractArray{T},
    y::AbstractVector,
    kernel::Kernel,
    likelihood::Union{TLikelihood,Distribution},
    inference::TInference;
    verbose::Int = 0,
    optimiser = ADAM(0.01),
    atfrequency::Integer = 1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean} = ZeroMean(),
    ArrayType::UnionAll = Vector,
) where {T<:Real,TLikelihood<:Likelihood,TInference<:SamplingInference}

    X = if X isa AbstractVector
        reshape(X, :, 1)
    else
        X
    end
    y, nLatent, likelihood = check_data!(X, y, likelihood)
    @assert inference isa SamplingInference "The inference object should be of type `SamplingInference` : either `GibbsSampling` or `HMCSampling`"
    @assert !isa(likelihood,GaussianLikelihood) "For a Gaussian Likelihood you should directly use the `GP` model or the `SVGP` model for large datasets"
    @assert implemented(likelihood, inference) "The $likelihood is not compatible or implemented with the $inference"

    nFeatures = nSamples = size(X, 1)
    nDim = size(X, 2)
    if isa(optimiser, Bool)
        optimiser = optimiser ? ADAM(0.01) : nothing
    end

    if typeof(mean) <: Real
        mean = ConstantMean(mean)
    elseif typeof(mean) <: AbstractVector{<:Real}
        mean = EmpiricalMean(mean)
    end

    latentf = ntuple(_ -> _MCGP{T}(nFeatures, kernel, mean), nLatent)

    likelihood =
        init_likelihood(likelihood, inference, nLatent, nSamples, nFeatures)
    inference = tuple_inference(inference, nLatent, nSamples, nSamples)
    inference.xview = view(X, :, :)
    inference.yview = view_y(likelihood, y, 1:nSamples)
    MCGP{T,TLikelihood,typeof(inference),nLatent}(
        X,
        y,
        nFeatures,
        nDim,
        nFeatures,
        nLatent,
        latentf,
        likelihood,
        inference,
        verbose,
        atfrequency,
        false,
    )
end

function Base.show(io::IO,model::MCGP{T,<:Likelihood,<:Inference}) where {T}
    print(io,"Monte Carlo Gaussian Process with a $(model.likelihood) sampled via $(model.inference) ")
end

get_f(model::MCGP) = getproperty.(model.f,:f)
get_y(model::MCGP) = model.inference.yview
get_Z(model::MCGP) = [model.inference.xview]
objective(model::MCGP{T}) where {T} = NaN

@traitimpl IsFull{MCGP}
