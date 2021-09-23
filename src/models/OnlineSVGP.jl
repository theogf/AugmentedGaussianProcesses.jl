mutable struct OnlineSVGP{
    T<:Real,TLikelihood<:AbstractLikelihood,TInference<:AbstractInference,N
} <: AbstractGPModel{T,TLikelihood,TInference,N}
    f::NTuple{N,OnlineVarLatent{T}}
    likelihood::TLikelihood
    inference::TInference
    verbose::Int
    atfrequency::Int
    trained::Bool
end

"""
    OnlineSVGP(args...; kwargs...)

Online Sparse Variational Gaussian Process

## Arguments 
- `kernel::Kernel` : Covariance function, can be any kernel from KernelFunctions.jl
- `likelihood` : Likelihood of the model. For compatibilities, see [`Likelihood Types`](@ref likelihood_user)
- `inference` : Inference for the model, see the [`Compatibility Table`](@ref compat_table))
- `Zalg` : Algorithm selecting how inducing points are selected

## Keywords arguments
- `verbose` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
- `optimiser` : Optimiser used for the kernel parameters. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `ADAM(0.001)`
- `atfrequency::Int=1` : Choose how many variational parameters iterations are between hyperparameters optimization
- `mean=ZeroMean()` : PriorMean object, check the documentation on it [`MeanPrior`](@ref meanprior)
- `Zoptimiser` : Optimiser for inducing points locations
- `T::DataType=Float64` : Hint for what the type of the data is going to be.
"""
function OnlineSVGP(
    kernel::Kernel,
    likelihood::AbstractLikelihood,
    inference::AbstractInference,
    Zalg::InducingPoints.OnIPSA=OIPS(0.9);
    verbose::Integer=0,
    optimiser=ADAM(0.01),
    atfrequency::Integer=1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
    Zoptimiser=nothing,
    T::DataType=Float64,
)
    inference isa AnalyticVI || error("The inference object should be of type `AnalyticVI`")
    implemented(likelihood, inference) ||
        error("The ", likelihood, " is not compatible or implemented with the ", inference)

    if isa(optimiser, Bool)
        optimiser = optimiser ? ADAM(0.01) : nothing
    end

    if typeof(mean) <: Real
        mean = ConstantMean(mean)
    elseif typeof(mean) <: AbstractVector{<:Real}
        mean = EmpiricalMean(mean)
    end

    num_latent = n_latent(likelihood)
    latentf = ntuple(num_latent) do _
        return OnlineVarLatent(T, 0, [], Zalg, kernel, mean, optimiser, Zoptimiser)
    end
    # inference.nIter = 1
    return OnlineSVGP{T,typeof(likelihood),typeof(inference),num_latent}(
        latentf, likelihood, inference, verbose, atfrequency, false
    )
end

function Base.show(io::IO, model::OnlineSVGP) where {T}
    return print(
        io,
        "Online Variational Gaussian Process with a ",
        likelihood(model),
        " infered by ",
        inference(model),
    )
end

@traitimpl IsSparse{OnlineSVGP}

Zviews(m::OnlineSVGP) = Zview.(m.f)
objective(m::OnlineSVGP, state, y) = ELBO(m, state, y)
