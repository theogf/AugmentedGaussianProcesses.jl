"""
    SVGP(args...; kwargs...)

Sparse Variational Gaussian Process

## Arguments
- `X::AbstractArray` : Input features, if `X` is a matrix the choice of colwise/rowwise is given by the `obsdim` keyword
- `y::AbstractVector` : Output labels
- `kernel::Kernel` : Covariance function, can be any kernel from KernelFunctions.jl
- `likelihood` : Likelihood of the model. For compatibilities, see [`Likelihood Types`](@ref likelihood_user)
- `inference` : Inference for the model, see the [`Compatibility Table`](@ref compat_table))
- `nInducingPoints/Z` : number of inducing points, or `AbstractVector` object

## Keyword arguments
- `verbose` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
- `optimiser` : Optimiser used for the kernel parameters. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `ADAM(0.001)`
- `atfrequency::Int=1` : Choose how many variational parameters iterations are between hyperparameters optimization
- `mean=ZeroMean()` : PriorMean object, check the documentation on it [`MeanPrior`](@ref meanprior)
- `Zoptimiser` : Optimiser for inducing points locations
- `obsdim::Int=1` : Dimension of the data. 1 : X ∈ DxN, 2: X ∈ NxD
"""
mutable struct SVGP{
    T<:Real,
    TLikelihood<:AbstractLikelihood,
    TInference<:AbstractInference,
    N,
} <: AbstractGPModel{T,TLikelihood,TInference,N}
    f::NTuple{N,SparseVarLatent{T}}
    likelihood::TLikelihood
    inference::TInference
    verbose::Int
    atfrequency::Int
    trained::Bool
end

function SVGP(
    kernel::Kernel,
    likelihood::AbstractLikelihood,
    inference::AbstractInference,
    Z::AbstractVector;
    verbose::Int=0,
    optimiser=ADAM(0.01),
    atfrequency::Int=1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
    Zoptimiser=nothing,
    T::DataType=Float64,
)
    inference isa VariationalInference || error(
        "The inference object should be of type `VariationalInference` : either `AnalyticVI` or `NumericalVI`",
    )
    implemented(likelihood, inference) ||
        error("The $likelihood is not compatible or implemented with the $inference")

    if isa(optimiser, Bool)
        optimiser = optimiser ? ADAM(0.001) : nothing
    end

    Zoptimiser = if Zoptimiser isa Bool
        Zoptimiser ? ADAM(0.001) : nothing
    else
        Zoptimiser
    end

    if typeof(mean) <: Real
        mean = ConstantMean(mean)
    elseif typeof(mean) <: AbstractVector{<:Real}
        mean = EmpiricalMean(mean)
    end

    latentf = ntuple(
        _ -> SparseVarLatent(T, Z, kernel, mean, optimiser, Zoptimiser),
        nLatent,
    )

    model = SVGP{T,typeof(likelihood),typeof(inference),typeof(data),nLatent}(
        data,
        latentf,
        likelihood,
        inference,
        verbose,
        atfrequency,
        false,
    )
    if isa(optimiser, ALRSVI)
        init!(model)
    end
    return model
end

function Base.show(io::IO, model::SVGP)
    return print(
        io,
        "Sparse Variational Gaussian Process with a $(likelihood(model)) infered by $(inference(model)) ",
    )
end

Zviews(m::SVGP) = Zview.(m.f)
objective(m::SVGP) = ELBO(m)

@traitimpl IsSparse{SVGP}
