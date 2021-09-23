"""
    VGP(args...; kwargs...)

Variational Gaussian Process

## Arguments
- `X::AbstractArray` : Input features, if `X` is a matrix the choice of colwise/rowwise is given by the `obsdim` keyword
- `y::AbstractVector` : Output labels
- `kernel::Kernel` : Covariance function, can be any kernel from KernelFunctions.jl
- `likelihood` : Likelihood of the model. For compatibilities, see [`Likelihood Types`](@ref likelihood_user)
- `inference` : Inference for the model, see the [`Compatibility Table`](@ref compat_table))

## Keyword arguments
- `verbose` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
- `optimiser` : Optimiser used for the kernel parameters. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `ADAM(0.001)`
- `atfrequency::Int=1` : Choose how many variational parameters iterations are between hyperparameters optimization
- `mean=ZeroMean()` : PriorMean object, check the documentation on it [`MeanPrior`](@ref meanprior)
- `obsdim::Int=1` : Dimension of the data. 1 : X ∈ DxN, 2: X ∈ NxD
"""
mutable struct VGP{
    T<:Real,
    TLikelihood<:AbstractLikelihood,
    TInference<:AbstractInference,
    TData<:AbstractDataContainer,
    N,
} <: AbstractGPModel{T,TLikelihood,TInference,N}
    data::TData # Data container
    f::NTuple{N,VarLatent{T}} # Vector of latent GPs
    likelihood::TLikelihood
    inference::TInference
    verbose::Int #Level of printing information
    atfrequency::Int
    trained::Bool
end

function VGP(
    X::AbstractArray,
    y::AbstractVector,
    kernel::Kernel,
    likelihood::AbstractLikelihood,
    inference::AbstractInference;
    verbose::Int=0,
    optimiser=ADAM(0.01),
    atfrequency::Integer=1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
    obsdim::Int=1,
)
    X, T = wrap_X(X, obsdim)
    y = check_data!(y, likelihood)

    inference isa VariationalInference || error(
        "The inference object should be of type `VariationalInference` : either `AnalyticVI` or `NumericalVI`",
    )
    !isa(likelihood, GaussianLikelihood) || error(
        "For a Gaussian Likelihood you should directly use the `GP` model or the `SVGP` model for large datasets",
    )
    implemented(likelihood, inference) ||
        error("The $likelihood is not compatible or implemented with the $inference")

    data = wrap_data(X, y)

    if isa(optimiser, Bool)
        optimiser = optimiser ? ADAM(0.01) : nothing
    end

    n_feature = n_sample(data)

    if typeof(mean) <: Real
        mean = ConstantMean(mean)
    elseif typeof(mean) <: AbstractVector{<:Real}
        mean = EmpiricalMean(mean)
    end

    latentf = ntuple(n_latent(likelihood)) do _
        VarLatent(T, n_feature, kernel, mean, optimiser)
    end

    return VGP{T,typeof(likelihood),typeof(inference),typeof(data),n_latent(likelihood)}(
        data, latentf, likelihood, inference, verbose, atfrequency, false
    )
end

function Base.show(io::IO, model::VGP)
    return print(
        io,
        "Variational Gaussian Process with a $(likelihood(model)) infered by $(inference(model)) ",
    )
end

Zviews(m::VGP) = (input(m.data),)
objective(m::VGP, state, y) = ELBO(m, state, y)

@traitimpl IsFull{VGP}
