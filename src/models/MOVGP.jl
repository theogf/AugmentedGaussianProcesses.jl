"""
Class for multi-output variational Gaussian Processes based on the Linear Coregionalization Model (LCM)

```julia
MOVGP(
    X::Union{AbstractArray{T}, AbstractVector{<:AbstractArray{T}}},
    y::AbstractVector{<:AbstractArray},
    kernel::Union{Kernel, AbstractVector{<:Kernel}},
    likelihood::Union{TLikelihood, AbstractVector{<:TLikelihood}},
    inference::TInference,
    nLatent::Int;
    verbose::Int = 0,
    optimiser = ADAM(0.01),
    atfrequency::Int = 1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean} = ZeroMean(),
    variance::Real = 1.0,
    Aoptimiser = ADAM(0.01),
    ArrayType::UnionAll = Vector,
) where {T<:Real,TLikelihood<:Likelihood,TInference<:Inference}
```

Argument list :

**Mandatory arguments**
 - `X` : input features, should be a matrix NxD where N is the number of observation and D the number of dimension
 - `y` : input labels, can be either a vector of labels for multiclass and single output or a matrix for multi-outputs (note that only one likelihood can be applied)
 - `kernel` : covariance function, can be either a single kernel or a collection of kernels for multiclass and multi-outputs models
 - `likelihood` : likelihood of the model, currently implemented : Gaussian, Student-T, Laplace, Bernoulli (with logistic link), Bayesian SVM, Multiclass (softmax or logistic-softmax) see [`Likelihood`](@ref likelihood_user)
 - `inference` : inference for the model, can be analytic, numerical or by sampling, check the model documentation to know what is available for your likelihood see the [`Compatibility table`](@ref compat_table)
**Optional arguments**
 - `verbose` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
- `optimiser` : Optimiser used for the kernel parameters. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `ADAM(0.001)`
 - `atfrequency` : Choose how many variational parameters iterations are between hyperparameters optimization
 - `mean` : PriorMean object, check the documentation on it [`MeanPrior`](@ref meanprior)
 - `IndependentPriors` : Flag for setting independent or shared parameters among latent GPs
 - `ArrayType` : Option for using different type of array for storage (allow for GPU usage)
"""
mutable struct MOVGP{
    T<:Real,
    TLikelihood<:Likelihood{T},
    TInference<:Inference,
    TData<:AbstractDataContainer,
    N,
    Q,
} <: AbstractGP{T,TLikelihood,TInference,N}
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
    X::Union{AbstractVector,AbstractVector{<:AbstractArray}},
    y::AbstractVector{<:AbstractArray},
    kernel::Union{Kernel,AbstractVector{<:Kernel}},
    likelihood::Union{TLikelihood,AbstractVector{<:TLikelihood}},
    inference::Inference,
    nLatent::Int;
    verbose::Int = 0,
    optimiser = ADAM(0.01),
    atfrequency::Int = 1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean} = ZeroMean(),
    variance::Real = 1.0,
    Aoptimiser = ADAM(0.01),
    ArrayType::UnionAll = Vector,
) where {TLikelihood<:Likelihood}

    @assert length(y) > 0 "y should not be an empty vector"
    nTask = length(y)

    X, T = wrap_X(X)
    n_task = length(y)

    likelihoods = if likelihood isa Likelihood
        likelihoods = [deepcopy(likelihood) for _ in 1:n_task]
    else
        likelihood
    end

    nf_per_task = zeros(Int64, nTask)
    corrected_y = Vector(undef, nTask)
    for i = 1:nTask
        corrected_y[i], nf_per_task[i], likelihoods[i] =
            check_data!(y[i], likelihoods[i])
    end

    inference isa AnalyticVI || error("The inference object should be of type `AnalyticVI`")
    all(implemented.(likelihood, Ref(inference))) || error("One (or all) of the likelihoods:  $likelihoods are not compatible or implemented with $inference")

    data = wrap_data(X, corrected_y)

    nFeatures = nSamples(data)

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
    elseif kernel isa AbstractVector{<:Kernel}
        length(kernel) == nLatent || error("Number of kernels should be equal to the number of tasks")
        kernel
    end
    nKernel = length(kernel)

    latent_f = ntuple(
        i -> _VGP{T}(
            nFeatures,
            kernel[mod(i, nKernel)+1],
            mean,
            optimiser,
        ),
        nLatent,
    )

    A = [
        [
            randn(nLatent) |> x -> x / sqrt(sum(abs2, x))
            for i = 1:nf_per_task[j]
        ] for j = 1:nTask
    ]

    likelihoods .=
        init_likelihood.(
            likelihoods,
            inference,
            nf_per_task,
            nFeatures,
            nFeatures,
        )
    xview = view_x(data, :)
    yview = view_y(likelihood, data, 1:nSamples(data))
    inference =
        tuple_inference(inference, nLatent, nFeatures, nSamples(data), nSamples(data), xview, yview)

    return MOVGP{T,TLikelihood,typeof(inference),nTask,nLatent}(
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
    # if isa(inference.optimizer,ALRSVI)
    # init!(model.inference,model)
    # end
    # return model
end

function Base.show(io::IO, model::MOVGP{T,<:Likelihood,<:Inference}) where {T}
    print(
        io,
        "Multioutput Variational Gaussian Process with the likelihoods $(model.likelihood) infered by $(model.inference) ",
    )
end

@traitimpl IsMultiOutput{MOVGP}
@traitimpl IsFull{MOVGP}


nOutput(m::MOVGP{<:Real,<:Likelihood,<:Inference,N,Q}) where {N,Q} = Q
Zviews(m::MOVGP) = [input(m)]
objective(m::MOVGP) = ELBO(m)
