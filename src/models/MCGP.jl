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
mutable struct MCGP{
    T<:Real,
    TLikelihood<:Likelihood{T},
    TInference<:Inference{T},
    TData<:AbstractDataContainer,
    N,
} <: AbstractGP{T,TLikelihood,TInference,N}
    data::TData
    f::NTuple{N,SampledLatent{T}} # Vector of latent GPs
    likelihood::TLikelihood
    inference::TInference
    verbose::Int64 #Level of printing information
    atfrequency::Int64
    trained::Bool
end


function MCGP(
    X::AbstractArray{<:Real},
    y::AbstractVector,
    kernel::Kernel,
    likelihood::Union{Likelihood,Distribution},
    inference::Inference;
    verbose::Int = 0,
    optimiser = ADAM(0.01),
    atfrequency::Integer = 1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean} = ZeroMean(),
    obsdim::Int = 1,
)
    X, T = wrap_X(X, obsdim)
    y, nLatent, likelihood = check_data!(y, likelihood)

    inference isa SamplingInference || error("The inference object should be of type `SamplingInference` : either `GibbsSampling` or `HMCSampling`")
    !isa(likelihood, GaussianLikelihood) ||  error("For a Gaussian Likelihood you should directly use the `GP` model or the `SVGP` model for large datasets")
    implemented(likelihood, inference) || error("The $likelihood is not compatible or implemented with the $inference")
    !isa(likelihood, Distribution) || error("Using Distributions.jl distributions is unfortunately not yet implemented")
    data = wrap_data(X, y)
    nFeatures = nSamples(data)

    if isa(optimiser, Bool)
        optimiser = optimiser ? ADAM(0.01) : nothing
    end

    if typeof(mean) <: Real
        mean = ConstantMean(mean)
    elseif typeof(mean) <: AbstractVector{<:Real}
        mean = EmpiricalMean(mean)
    end

    latentf = ntuple(_ -> SampledLatent(T, nFeatures, kernel, mean), nLatent)

    likelihood =
        init_likelihood(likelihood, inference, nLatent, nSamples(data))
    xview = view_x(data, 1:nSamples(data))
    yview = view_y(likelihood, data, 1:nSamples(data))
    inference =
        tuple_inference(inference, nLatent, nSamples(data), nSamples(data), nSamples(data), xview, yview)
    MCGP{T,typeof(likelihood),typeof(inference),typeof(data),nLatent}(
        data,
        latentf,
        likelihood,
        inference,
        verbose,
        atfrequency,
        false,
    )
end

function Base.show(io::IO, model::MCGP{T,<:Likelihood,<:Inference}) where {T}
    print(
        io,
        "Monte Carlo Gaussian Process with a $(model.likelihood) sampled via $(model.inference) ",
    )
end

Zviews(model::MCGP) = [input(model)]
objective(model::MCGP{T}) where {T} = NaN

@traitimpl IsFull{MCGP}
