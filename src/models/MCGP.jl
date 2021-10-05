""" 
    MCGP(args...; kwargs...)

Monte-Carlo Gaussian Process

## Arguments
- `X::AbstractArray` : Input features, if `X` is a matrix the choice of colwise/rowwise is given by the `obsdim` keyword
- `y::AbstractVector` : Output labels
- `kernel::Kernel` : Covariance function, can be any kernel from KernelFunctions.jl
- `likelihood` : Likelihood of the model. For compatibilities, see [`Likelihood Types`](@ref likelihood_user)
- `inference` : Inference for the model, at the moment only [`GibbsSampling`](@ref) is available (see the [`Compatibility Table`](@ref compat_table))

## Keyword arguments
- `verbose::Int` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
- `optimiser` : Optimiser used for the kernel parameters. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `ADAM(0.001)`
- `atfrequency::Int=1` : Choose how many variational parameters iterations are between hyperparameters optimization
- `mean=ZeroMean()` : PriorMean object, check the documentation on it [`MeanPrior`](@ref meanprior)
- `obsdim::Int=1` : Dimension of the data. 1 : X ∈ DxN, 2: X ∈ NxD 
"""
mutable struct MCGP{
    T<:Real,
    TLikelihood<:AbstractLikelihood,
    TInference<:AbstractInference{T},
    TData<:AbstractDataContainer,
    N,
} <: AbstractGPModel{T,TLikelihood,TInference,N}
    data::TData
    f::NTuple{N,SampledLatent{T}} # Vector of latent GPs
    likelihood::TLikelihood
    inference::TInference
    verbose::Int64 #Level of printing information
    atfrequency::Int64
    trained::Bool
end

function MCGP(
    X::AbstractArray,
    y::AbstractVector,
    kernel::Kernel,
    likelihood::Union{AbstractLikelihood,Distribution},
    inference::AbstractInference;
    verbose::Int=0,
    optimiser=ADAM(0.01),
    atfrequency::Integer=1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
    obsdim::Int=1,
)
    X, T = wrap_X(X, obsdim)
    y = check_data!(y, likelihood)

    inference isa SamplingInference || error(
        "The inference object should be of type `SamplingInference` : either `GibbsSampling` or `HMCSampling`",
    )
    !isa(likelihood, GaussianLikelihood) || error(
        "For a Gaussian Likelihood you should directly use the `GP` model or the `SVGP` model for large datasets",
    )
    implemented(likelihood, inference) ||
        error("The $likelihood is not compatible or implemented with the $inference")
    !isa(likelihood, Distribution) ||
        error("Using Distributions.jl distributions is unfortunately not yet implemented")
    data = wrap_data(X, y)
    n_feature = n_sample(data)

    if isa(optimiser, Bool)
        optimiser = optimiser ? ADAM(0.01) : nothing
    end

    if typeof(mean) <: Real
        mean = ConstantMean(mean)
    elseif typeof(mean) <: AbstractVector{<:Real}
        mean = EmpiricalMean(mean)
    end

    latentf = ntuple(n_latent(likelihood)) do _
        return SampledLatent(T, n_feature, kernel, mean)
    end

    return MCGP{T,typeof(likelihood),typeof(inference),typeof(data),n_latent(likelihood)}(
        data, latentf, likelihood, inference, verbose, atfrequency, false
    )
end

function Base.show(io::IO, model::MCGP)
    return print(
        io,
        "Monte Carlo Gaussian Process with a $(likelihood(model)) sampled via $(inference(model)) ",
    )
end

Zviews(model::MCGP) = [input(model.data)]
objective(::MCGP{T}, ::Any, ::Any) where {T} = NaN

@traitimpl IsFull{MCGP}
