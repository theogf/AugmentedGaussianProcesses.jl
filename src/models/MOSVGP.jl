"""
Class for multi-output sparse variational Gaussian Processes based on the Linear Coregionalization Model (LCM)

```julia
MOSVGP(X::AbstractArray{T},y::AbstractVector{AbstractArray{T}},kernel::Kernel,
    likelihood::AbstractVector{Likelihoods},inference::InferenceType, nInducingPoints::Int;
    verbose::Int=0,optimiser=ADAM(0.001),atfrequency::Int=1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
    Zoptimiser=false,
    ArrayType::UnionAll=Vector)
```

Argument list :

**Mandatory arguments**
 - `X` : input features, should be a matrix N×D where N is the number of observation and D the number of dimension
 - `y` : input labels, can be either a vector of labels for multiclass and single output or a matrix for multi-outputs (note that only one likelihood can be applied)
 - `kernel` : covariance function, can be either a single kernel or a collection of kernels for multiclass and multi-outputs models
 - `likelihood` : likelihood of the model, currently implemented : Gaussian, Student-T, Laplace, Bernoulli (with logistic link), Bayesian SVM, Multiclass (softmax or logistic-softmax) see [`Likelihood`](@ref likelihood_user)
 - `inference` : inference for the model, can be analytic, numerical or by sampling, check the model documentation to know what is available for your likelihood see the [`Compatibility table`](@ref compat_table)
 - `nInducingPoints` : number of inducing points
**Optional arguments**
 - `verbose` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
- `optimiser` : Optimiser used for the kernel parameters. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `ADAM(0.001)`
 - `atfrequency` : Choose how many variational parameters iterations are between hyperparameters optimization
 - `mean` : PriorMean object, check the documentation on it [`MeanPrior`](@ref meanprior)
 - `IndependentPriors` : Flag for setting independent or shared parameters among latent GPs
- `Zoptimiser` : Optimiser used for inducing points locations. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `ADAM(0.001)`
 - `ArrayType` : Option for using different type of array for storage (allow for GPU usage)
"""
mutable struct MOSVGP{
    T<:Real,
    TLikelihood<:Likelihood{T},
    TInference<:Inference,
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
    X::Union{AbstractArray{T},AbstractVector{<:AbstractArray{T}}},
    y::AbstractVector{<:AbstractVector},
    kernel::Union{Kernel,AbstractVector{<:Kernel}},
    likelihood::Union{TLikelihood,AbstractVector{<:TLikelihood}},
    inference::TInference,
    nLatent::Int,
    nInducingPoints::Union{
        Int,
        AbstractInducingPoints,
        AbstractVector{<:AbstractInducingPoints},
    };
    verbose::Int = 0,
    atfrequency::Int = 1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean} = ZeroMean(),
    variance::Real = 1.0,
    optimiser = ADAM(0.01),
    Aoptimiser = ADAM(0.01),
    Zoptimiser = false,
    ArrayType::UnionAll = Vector,
) where {T<:Real,TLikelihood<:Likelihood,TInference<:Inference}

    @assert length(y) > 0 "y should not be an empty vector"
    nTask = length(y)

    X = wrap_X_multi(X, nTask)
    nX = length(X)

    likelihoods = if likelihood isa Likelihood
        likelihoods = [deepcopy(likelihood) for _ = 1:nTask]
    else
        likelihood
    end

    nf_per_task = zeros(Int64, nTask)
    corrected_y = Vector(undef, nTask)
    for i = 1:nTask
        corrected_y[i], nf_per_task[i], likelihoods[i] =
            check_data!(X[mod(i, nX)+1], y[i], likelihoods[i])
    end

    @assert inference isa AnalyticVI "The inference object should be of type `AnalyticVI`"
    @assert all(implemented.(likelihood, Ref(inference))) "The $likelihood is not compatible or implemented with the $inference"

    nSamples = size.(X, 1)
    nDim = size.(X, 2)

    if isa(optimiser, Bool)
        optimiser = optimiser ? ADAM(0.01) : nothing
    end

    if isa(Aoptimiser, Bool)
        Aoptimiser = Aoptimiser ? ADAM(0.01) : nothing
    end

    kernel = if kernel isa Kernel
        [kernel]
    else
        @assert length(kernel) == nLatent "Number of kernels should be equal to the number of tasks"
        kernel
    end
    nKernel = length(kernel)
    nInducingPoints = nInducingPoints isa AbstractInducingPoints ?
        [deepcopy(nInducingPoints) for _ = 1:nLatent] : nInducingPoints
    Z = init_Z.(nInducingPoints, nSamples, X, y, kernel, Ref(Zoptimiser))

    nFeatures = size.(Z, 1)

    if typeof(mean) <: Real
        mean = ConstantMean(mean)
    elseif typeof(mean) <: AbstractVector{<:Real}
        mean = EmpiricalMean(mean)
    end

    _nMinibatch = nSamples
    if isStochastic(inference)
        @assert all(0 .< nMinibatch(inference) .< nSamples) "The size of mini-batch $(nMinibatch(inference)) is incorrect (negative or bigger than number of samples), please set nMinibatch correctly in the inference object"
        _nMinibatch = inference.nMinibatch
    end

    @show nFeatures, _nMinibatch
    latent_f = ntuple(
        i -> _SVGP{T}(
            nFeatures[mod(i, nLatent)+1],
            _nMinibatch[1],
            Z[mod(i, nLatent)+1],
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
            _nMinibatch[1:1],
            nFeatures,
        )
    inference = tuple_inference(
        inference,
        nLatent,
        nFeatures,
        nSamples,
        _nMinibatch[1:1],
    )
    inference.xview = view.(X, collect.(range.(1, _nMinibatch, step = 1)), :)
    inference.yview = view(y, :)

    model = MOSVGP{T,TLikelihood,typeof(inference),nTask,nLatent}(
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
    # if isa(inference.optimizer,ALRSVI)
    # init!(model.inference,model)
    # end
    # return model
end

function Base.show(io::IO, model::MOSVGP{T,<:Likelihood,<:Inference}) where {T}
    print(
        io,
        "Multioutput Sparse Variational Gaussian Process with the likelihoods $(model.likelihood) infered by $(model.inference) ",
    )
end

@traitimpl IsMultiOutput{MOSVGP}

nOutput(m::MOSVGP{<:Real,<:Likelihood,<:Inference,N,Q}) where {N, Q} = Q
get_X(m::MOSVGP) = m.X
get_Z(m::MOSVGP) = get_Z.(m.f)
get_Z(m::MOSVGP, i::Int) = get_Z(m.f[i])
objective(m::MOSVGP) = ELBO(m)
