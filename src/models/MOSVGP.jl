"""
    MOSVGP(args...; kwargs...)

Multi-Output Sparse Variational Gaussian Process

## Arguments
- `kernel::Union{Kernel,AbstractVector{<:Kernel}` : covariance function or vector of covariance functions, can be either a single kernel or a collection of kernels for multiclass and multi-outputs models
- `likelihoods::Union{AbstractLikelihood,Vector{<:Likelihood}` : Likelihood or vector of likelihoods of the model. For compatibilities, see [`Likelihood Types`](@ref likelihood_user)
- `inference` : Inference for the model, for compatibilities see the [`Compatibility Table`](@ref compat_table))
- `nLatent::Int` : Number of latent GPs
- `nInducingPoints` : number of inducing points, or collection of inducing points locations

## Keyword arguments
- `verbose::Int` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
- `optimiser` : Optimiser used for the kernel parameters. Should be an Optimiser object from the [Optimisers.jl](https://github.com/FluxML/Optimisers.jl) library. Default is `ADAM(0.001)`
- `Zoptimiser` : Optimiser used for the inducing points locations
- `Aoptimiser` : Optimiser used for the mixing parameters.
- `atfrequency::Int=1` : Choose how many variational parameters iterations are between hyperparameters optimization
- `mean=ZeroMean()` : PriorMean object, check the documentation on it [`MeanPrior`](@ref meanprior)
- `obsdim::Int=1` : Dimension of the data. 1 : X ∈ DxN, 2: X ∈ NxD
"""
mutable struct MOSVGP{
    T<:Real,
    TLikelihood,
    TInference<:AbstractInference,
    N, # Number of tasks
    Q, # Number of latent GPs
} <: AbstractGPModel{T,AbstractLikelihood,TInference,N}
    nf_per_task::NTuple{N,Int}
    f::NTuple{Q,SparseVarLatent}
    likelihood::TLikelihood
    inference::TInference
    A::Vector{Vector{Vector{T}}}
    A_opt::Any
    verbose::Int
    atfrequency::Int
    trained::Bool
end

function MOSVGP(
    kernel::Union{Kernel,AbstractVector{<:Kernel}},
    likelihoods::Union{
        AbstractVector{<:AbstractLikelihood},Tuple{Vararg{<:AbstractLikelihood}}
    },
    inference::AbstractInference,
    Zs::AbstractVector;
    verbose::Int=0,
    atfrequency::Int=1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
    optimiser=ADAM(0.01),
    Aoptimiser=ADAM(0.01),
    Zoptimiser=false,
    T::DataType=Float64,
)
    likelihoods = likelihoods isa AbstractVector ? tuple(likelihoods...) : likelihoods

    n_task = length(likelihoods)
    nf_per_task = n_latent.(likelihoods)

    inference isa AnalyticVI || error("The inference object should be of type `AnalyticVI`")
    all(implemented.(likelihoods, Ref(inference))) || error(
        "One (or more) of the likelihoods $likelihoods are not compatible or implemented with the $inference",
    )

    if mean isa Real
        mean = ConstantMean(mean)
    elseif mean isa AbstractVector{<:Real}
        mean = EmpiricalMean(mean)
    end

    if isa(optimiser, Bool)
        optimiser = optimiser ? ADAM(0.01) : nothing
    end

    if isa(Zoptimiser, Bool)
        Zoptimiser = Zoptimiser ? ADAM(0.001) : nothing
    end

    if isa(Aoptimiser, Bool)
        Aoptimiser = Aoptimiser ? ADAM(0.01) : nothing
    end

    kernel = if kernel isa Kernel
        (kernel,)
    else
        length(kernel) == n_task ||
            error("Number of kernels should be equal to the number of tasks")
        kernel
    end

    n_kernel = length(kernel)

    num_latent = length(Zs)

    latent_f = ntuple(num_latent) do i
        SparseVarLatent(T, Zs[i], kernel[mod1(i, n_kernel)], mean, optimiser, Zoptimiser)
    end

    function normalize(x)
        return x / sqrt(sum(abs2, x))
    end
    A = [[normalize(randn(T, num_latent)) for i in 1:nf_per_task[j]] for j in 1:n_task]

    return MOSVGP{T,typeof(likelihoods),typeof(inference),n_task,num_latent}(
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

n_output(::MOSVGP{T,L,I,N,Q}) where {T,L,I,N,Q} = Q
Zviews(m::MOSVGP) = Zview.(m.f)
objective(m::MOSVGP, state, y) = ELBO(m, state, y)
