"""
    MOSVGP(args...; kwargs...)

Multi-Output Sparse Variational Gaussian Process

## Arguments
- `X::AbstractArray` : : Input features, if `X` is a matrix the choice of colwise/rowwise is given by the `obsdim` keyword
- `y::AbstractVector{<:AbstractVector}` : Output labels, each vector corresponds to one output dimension
- `kernel::Union{Kernel,AbstractVector{<:Kernel}` : covariance function or vector of covariance functions, can be either a single kernel or a collection of kernels for multiclass and multi-outputs models
- `likelihood::Union{AbstractLikelihood,Vector{<:Likelihood}` : Likelihood or vector of likelihoods of the model. For compatibilities, see [`Likelihood Types`](@ref likelihood_user)
- `inference` : Inference for the model, for compatibilities see the [`Compatibility Table`](@ref compat_table))
- `nLatent::Int` : Number of latent GPs
- `nInducingPoints` : number of inducing points, or collection of inducing points locations

## Keyword arguments
- `verbose::Int` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
- `optimiser` : Optimiser used for the kernel parameters. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `ADAM(0.001)`
- `Zoptimiser` : Optimiser used for the inducing points locations
- `Aoptimiser` : Optimiser used for the mixing parameters.
- `atfrequency::Int=1` : Choose how many variational parameters iterations are between hyperparameters optimization
- `mean=ZeroMean()` : PriorMean object, check the documentation on it [`MeanPrior`](@ref meanprior)
- `obsdim::Int=1` : Dimension of the data. 1 : X ∈ DxN, 2: X ∈ NxD
"""
mutable struct MOSVGP{
    T<:Real,
    TLikelihood<:AbstractLikelihood,
    TInference<:AbstractInference,
    TData<:AbstractDataContainer,
    N,
    Q,
} <: AbstractGP{T,TLikelihood,TInference,N}
    data::TData
    nFeatures::Vector{Int64} # Number of features of the GP (equal to number of points)
    nf_per_task::Vector{Int64}
    f::NTuple{Q,SparseVarLatent}
    likelihood::Vector{TLikelihood}
    inference::TInference
    A::Vector{Vector{Vector{T}}}
    A_opt::Any
    verbose::Int64
    atfrequency::Int64
    trained::Bool
end

function MOSVGP(
    X::AbstractArray,
    y::AbstractVector{<:AbstractVector},
    kernel::Union{Kernel,AbstractVector{<:Kernel}},
    likelihood::Union{AbstractLikelihood,AbstractVector{<:AbstractLikelihood}},
    inference::AbstractInference,
    nLatent::Int,
    nInducingPoints::Union{
        Int,AbstractInducingPoints,AbstractVector{<:AbstractInducingPoints}
    };
    verbose::Int=0,
    atfrequency::Int=1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
    variance::Real=1.0,
    optimiser=ADAM(0.01),
    Aoptimiser=ADAM(0.01),
    Zoptimiser=false,
    obsdim::Int=1,
)
    @assert length(y) > 0 "y should not be an empty vector"
    nTask = length(y)

    X, T = wrap_X(X, obsdim)

    likelihoods = if likelihood isa AbstractLikelihood
        likelihoods = [deepcopy(likelihood) for _ in 1:nTask]
    else
        likelihood
    end

    nf_per_task = zeros(Int64, nTask)
    corrected_y = Vector(undef, nTask)
    for i in 1:nTask
        corrected_y[i], nf_per_task[i], likelihoods[i] = check_data!(y[i], likelihoods[i])
    end

    inference isa AnalyticVI || error("The inference object should be of type `AnalyticVI`")
    all(implemented.(likelihood, Ref(inference))) ||
        error("The $likelihood is not compatible or implemented with the $inference")

    data = wrap_data(X, corrected_y)

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

    kernel = if kernel isa Kernel
        [kernel]
    else
        length(kernel) == nLatent ||
            error("Number of kernels should be equal to the number of tasks")
        kernel
    end
    nKernel = length(kernel)

    nInducingPoints = if nInducingPoints isa AbstractInducingPoints
        [deepcopy(nInducingPoints) for _ in 1:nLatent]
    else
        nInducingPoints
    end
    Z = init_Z.(nInducingPoints, Ref(Zoptimiser))

    nFeatures = size.(Z, 1)

    _nMinibatch = nSamples(data)
    if isStochastic(inference)
        0 < nMinibatch(inference) < nSamples || error(
            "The size of mini-batch $(nMinibatch(inference)) is incorrect (negative or bigger than number of samples), please set nMinibatch correctly in the inference object",
        )
        _nMinibatch = nMinibatch(inference)
    end

    latent_f = ntuple(
        i -> _SVGP{T}(
            nFeatures[i],
            _nMinibatch,
            Z[mod(i, nLatent) + 1],
            kernel[mod(i, nKernel) + 1],
            mean,
            optimiser,
        ),
        nLatent,
    )

    A = [
        [x -> x / sqrt(sum(abs2, x))(randn(T, nLatent)) for i in 1:nf_per_task[j]] for
        j in 1:nTask
    ]

    likelihoods .=
        init_likelihood.(likelihoods, inference, nf_per_task, _nMinibatch, nFeatures)
    xview = view_x(data, collect(range(1, _nMinibatch; step=1)))
    yview = view_y(likelihood, data, 1:nSamples(data))
    inference = tuple_inference(
        inference, nLatent, nFeatures, nSamples(data), _nMinibatch, xview, yview
    )

    return MOSVGP{T,eltype(likelihoods),typeof(inference),nTask,nLatent}(
        X,
        corrected_y,
        nSamples,
        nDim,
        nFeatures,
        nLatent,
        nX,
        nTask,
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

function Base.show(io::IO, model::MOSVGP)
    return print(
        io,
        "Multioutput Sparse Variational Gaussian Process with the likelihoods $(likelihood(model)) infered by $(inference(model)) ",
    )
end

@traitimpl IsMultiOutput{MOSVGP}

nOutput(::MOSVGP{<:Real,<:AbstractLikelihood,<:AbstractInference,N,Q}) where {N,Q} = Q
Zviews(m::MOSVGP) = Zview.(m.f)
objective(m::MOSVGP) = ELBO(m)
