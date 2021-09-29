"""
    MOVGP(args...; kwargs...)

Multi-Output Variational Gaussian Process

## Arguments
- `X::AbstractVector` : : Input features, if `X` is a matrix the choice of colwise/rowwise is given by the `obsdim` keyword
- `y::AbstractVector{<:AbstractVector}` : Output labels, each vector corresponds to one output dimension
- `kernel::Union{Kernel,AbstractVector{<:Kernel}` : covariance function or vector of covariance functions, can be either a single kernel or a collection of kernels for multiclass and multi-outputs models
- `likelihood::Union{AbstractLikelihood,Vector{<:Likelihood}` : Likelihood or vector of likelihoods of the model. For compatibilities, see [`Likelihood Types`](@ref likelihood_user)
- `inference` : Inference for the model, for compatibilities see the [`Compatibility Table`](@ref compat_table))
- `nLatent::Int` : Number of latent GPs

## Keyword arguments
- `verbose::Int` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
- `optimiser` : Optimiser used for the kernel parameters. Should be an Optimiser object from the [Optimisers.jl](https://github.com/FluxML/Optimisers.jl) library. Default is `ADAM(0.001)`
- `Aoptimiser` : Optimiser used for the mixing parameters.
- `atfrequency::Int=1` : Choose how many variational parameters iterations are between hyperparameters optimization
- `mean=ZeroMean()` : PriorMean object, check the documentation on it [`MeanPrior`](@ref meanprior)
- `obsdim::Int=1` : Dimension of the data. 1 : X ∈ DxN, 2: X ∈ NxD
"""
mutable struct MOVGP{
    T<:Real,
    TLikelihood<:AbstractLikelihood,
    TInference<:AbstractInference,
    TData<:AbstractDataContainer,
    N,
    Q,
} <: AbstractGPModel{T,TLikelihood,TInference,N}
    data::TData
    nf_per_task::Vector{Int64}
    f::NTuple{Q,VarLatent{T}}
    likelihood::Vector{TLikelihood}
    inference::TInference
    A::Vector{Vector{Vector{T}}}
    A_opt::Any
    verbose::Int64
    atfrequency::Int64
    trained::Bool
end

function MOVGP(
    X::AbstractVector,
    y::AbstractVector{<:AbstractArray},
    kernel::Union{Kernel,AbstractVector{<:Kernel}},
    likelihood::Union{AbstractLikelihood,AbstractVector{<:AbstractLikelihood}},
    inference::AbstractInference,
    nLatent::Int;
    verbose::Int=0,
    optimiser=ADAM(0.01),
    atfrequency::Int=1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
    Aoptimiser=ADAM(0.01),
    obsdim::Int=1,
)
    nTask = length(y)

    X, T = wrap_X(X, obsdim)
    n_task = length(y)

    likelihoods = if likelihood isa AbstractLikelihood
        likelihoods = [deepcopy(likelihood) for _ in 1:n_task]
    else
        likelihood
    end

    corrected_y = map(check_data!, y, likelihoods)
    nf_per_task = n_latent.(likelihoods)

    inference isa AnalyticVI || error("The inference object should be of type `AnalyticVI`")
    all(implemented.(likelihood, Ref(inference))) || error(
        "One (or all) of the likelihoods:  $likelihoods are not compatible or implemented with $inference",
    )

    data = wrap_modata(X, corrected_y)

    nFeatures = n_sample(data)

    if mean isa Real
        mean = ConstantMean(mean)
    elseif mean isa AbstractVector{<:Real}
        mean = EmpiricalMean(mean)
    end

    if isa(optimiser, Bool)
        optimiser = optimiser ? ADAM(0.01) : nothing
    end

    if isa(Aoptimiser, Bool)
        Aoptimiser = Aoptimiser ? ADAM(0.01) : nothing
    end

    kernels = if kernel isa Kernel
        [kernel]
    elseif kernel isa AbstractVector{<:Kernel}
        length(kernel) == nLatent ||
            error("Number of kernels should be equal to the number of tasks")
        kernel
    end
    n_kernel = length(kernels)

    latent_f = ntuple(nLatent) do i
        return VarLatent(T, nFeatures, kernels[mod(i, n_kernel) + 1], mean, optimiser)
    end
    function normalize(x)
        return x / sqrt(sum(abs2, x))
    end
    A = [[normalize(randn(T, nLatent)) for i in 1:nf_per_task[j]] for j in 1:nTask]
    @show data

    return MOVGP{T,eltype(likelihoods),typeof(inference),typeof(data),nTask,nLatent}(
        data,
        nf_per_task,
        latent_f,
        likelihoods,
        inference,
        A,
        Aoptimiser,
        verbose,
        atfrequency,
        false,
    )
end

function Base.show(io::IO, model::MOVGP)
    return print(
        io,
        "Multioutput Variational Gaussian Process with the likelihoods $(likelihood(model)) infered by $(inference(model)) ",
    )
end

@traitimpl IsMultiOutput{MOVGP}
@traitimpl IsFull{MOVGP}

n_output(::MOVGP{<:Real,<:AbstractLikelihood,<:AbstractInference,N,Q}) where {N,Q} = Q
Zviews(m::MOVGP) = (input(m),)
objective(m::MOVGP, state, y) = ELBO(m, state, y)
