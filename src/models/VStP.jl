"""
    VStP(args...; kwargs...)

Variational Student-T Process

## Arguments
- `X::AbstractArray` : Input features, if `X` is a matrix the choice of colwise/rowwise is given by the `obsdim` keyword
- `y::AbstractVector` : Output labels
- `kernel::Kernel` : Covariance function, can be any kernel from KernelFunctions.jl
- `likelihood` : Likelihood of the model. For compatibilities, see [`Likelihood Types`](@ref likelihood_user)
- `inference` : Inference for the model, see the [`Compatibility Table`](@ref compat_table))
- `ν::Real` : Number of degrees of freedom 

## Keyword arguments
- `verbose` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
- `optimiser` : Optimiser used for the kernel parameters. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `ADAM(0.001)`
- `atfrequency::Int=1` : Choose how many variational parameters iterations are between hyperparameters optimization
- `mean=ZeroMean()` : PriorMean object, check the documentation on it [`MeanPrior`](@ref meanprior)
- `obsdim::Int=1` : Dimension of the data. 1 : X ∈ DxN, 2: X ∈ NxD
"""
mutable struct VStP{
    T<:Real,
    TLikelihood<:AbstractLikelihood,
    TInference<:AbstractInference,
    TData<:AbstractDataContainer,
    N,
} <: AbstractGPModel{T,TLikelihood,TInference,N}
    data::TData
    f::NTuple{N,TVarLatent{T}}
    likelihood::TLikelihood
    inference::TInference
    verbose::Int64 #Level of printing information
    atfrequency::Int64
    trained::Bool
end

function VStP(
    X::AbstractArray,
    y::AbstractVector,
    kernel::Kernel,
    likelihood::AbstractLikelihood,
    inference::AbstractInference,
    ν::Real;
    verbose::Int=0,
    optimiser=ADAM(0.01),
    atfrequency::Int=1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
    obsdim::Int=1,
)
    X, T = wrap_X(X, obsdim)
    y = check_data!(y, likelihood)

    inference isa VariationalInference || error(
        "The inference object should be of type `VariationalInference` : either `AnalyticVI` or `NumericalVI`",
    )
    implemented(likelihood, inference) ||
        error("The $likelihood is not compatible or implemented with the $inference")

    data = wrap_data(X, y)

    ν > 1 || error("ν should be bigger than 1")

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
        return TVarLatent(T, ν, n_feature, kernel, mean, optimiser)
    end

    return VStP{T,typeof(likelihood),typeof(inference),typeof(data),n_latent(likelihood)}(
        data, latentf, likelihood, inference, verbose, atfrequency, false
    )
end

function Base.show(io::IO, model::VStP)
    return print(
        io,
        "Variational Student-T Process with a $(likelihood(model)) infered by $(inference(model)) ",
    )
end

function local_prior_updates!(model::VStP, state, X)
    kernel_matrices = state.kernel_matrices
    for (gp, k_mat) in zip(model.f, kernel_matrices)
        local_prior_updates!(gp, X, k_mat.K)
    end
    return state
end

function local_prior_updates!(gp::TVarLatent, X, K)
    prior(gp).l² =
        (
            prior(gp).ν +
            dim(gp) +
            invquad(K, mean(gp) - pr_mean(gp, X)) +
            trace_ABt(inv(K), cov(gp))
        ) / 2
    return prior(gp).χ = (prior(gp).ν + dim(gp)) / (prior(gp).ν .+ prior(gp).l²)
end

Zviews(m::VStP) = (input(m.data),)
objective(m::VStP, state, y) = ELBO(m, state, y)

@traitimpl IsFull{VStP}
