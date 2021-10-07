"""
    MOVGP(args...; kwargs...)

Multi-Output Variational Gaussian Process

## Arguments
- `X::AbstractVector` : : Input features, if `X` is a matrix the choice of colwise/rowwise is given by the `obsdim` keyword
- `y::AbstractVector{<:AbstractVector}` : Output labels, each vector corresponds to one output dimension
- `kernel::Union{Kernel,AbstractVector{<:Kernel}` : covariance function or vector of covariance functions, can be either a single kernel or a collection of kernels for multiclass and multi-outputs models
- `likelihood::Union{AbstractLikelihood,Vector{<:Likelihood}` : Likelihood or vector of likelihoods of the model. For compatibilities, see [`Likelihood Types`](@ref likelihood_user)
- `inference` : Inference for the model, for compatibilities see the [`Compatibility Table`](@ref compat_table))
- `num_latent::Int` : Number of latent GPs

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
    TLikelihood,
    TInference<:AbstractInference,
    TData<:AbstractDataContainer,
    N, # Number of taks
    Q,
} <: AbstractGPModel{T,AbstractLikelihood,TInference,N}
    data::TData
    nf_per_task::NTuple{N,Int}
    f::NTuple{Q,VarLatent{T}}
    likelihood::TLikelihood
    inference::TInference
    A::Vector{Vector{Vector{T}}}
    A_opt::Any
    verbose::Int
    atfrequency::Int
    trained::Bool
end

function MOVGP(
    X::AbstractVector,
    y::AbstractVector{<:AbstractArray},
    kernel::Union{Kernel,AbstractVector{<:Kernel}},
    likelihoods::Union{
        AbstractVector{<:AbstractLikelihood},Tuple{Vararg{<:AbstractLikelihood}}
    },
    inference::AbstractInference,
    num_latent::Int;
    verbose::Int=0,
    optimiser=ADAM(0.01),
    atfrequency::Int=1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
    Aoptimiser=ADAM(0.01),
    obsdim::Int=1,
    T::DataType=Float64,
)
    X, _ = wrap_X(X, obsdim)

    likelihoods = likelihoods isa AbstractVector ? tuple(likelihoods...) : likelihoods

    n_task = length(likelihoods)
    nf_per_task = n_latent.(likelihoods)

    corrected_y = map(check_data!, y, likelihoods)

    inference isa AnalyticVI || error("The inference object should be of type `AnalyticVI`")
    all(implemented.(likelihoods, Ref(inference))) || error(
        "One (or more) of the likelihoods $likelihoods are not compatible or implemented with $inference",
    )

    data = wrap_modata(X, corrected_y)

    n_feature = n_sample(data)

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
        (kernel,)
    else
        length(kernel) == num_latent ||
            error("Number of kernels should be equal to the number of tasks")
        kernel
    end
    n_kernel = length(kernels)

    latent_f = ntuple(num_latent) do i
        return VarLatent(T, n_feature, kernels[mod1(i, n_kernel)], mean, optimiser)
    end
    function normalize(x)
        return x / sqrt(sum(abs2, x))
    end
    A = [[normalize(randn(T, num_latent)) for i in 1:nf_per_task[j]] for j in 1:n_task]

    return MOVGP{T,typeof(likelihoods),typeof(inference),typeof(data),n_task,num_latent}(
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

n_output(::MOVGP{T,L,I,N,Q}) where {T,L,I,N,Q} = Q
Zviews(m::MOVGP) = (input(m.data),)
objective(m::MOVGP, state, y) = ELBO(m, state, y)
