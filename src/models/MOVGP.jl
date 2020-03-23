"""
Class for multi-output variational Gaussian Processes based on the Linear Coregionalization Model (LCM)

```julia
MOVGP(X::AbstractArray{T},y::AbstractVector{AbstractArray{T}},kernel::Kernel,
    likelihood::AbstractVector{Likelihoods},inference::InferenceType,
    verbose::Int=0,optimiser=ADAM(0.001),atfrequency::Int=1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
    ArrayType::UnionAll=Vector)
```

Argument list :

**Mandatory arguments**
 - `X` : input features, should be a vector of matrices N×D or one matrix NxD where N is the number of observation and D the number of dimension
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
mutable struct MOVGP{T<:Real,TLikelihood<:Likelihood{T},TInference<:Inference,N,Q} <: AbstractGP{T,TLikelihood,TInference,N}
    X::Vector{Matrix{T}} #Feature vectors
    y::Vector #Output (-1,1 for classification, real for regression, matrix for multiclass)
    nSamples::Int64 # Number of data points
    nDim::Int64 # Number of covariates per data point
    nFeatures::Int64 # Number of features of the GP (equal to number of points)
    nLatent::Int64 # Number of latent GPs
    nTask::Int64
    nf_per_task::Vector{Int64}
    f::NTuple{Q,_VGP}
    likelihood::Vector{TLikelihood}
    inference::TInference
    A::Vector{Vector{Vector{T}}}
    A_opt
    verbose::Int64
    atfrequency::Int64
    Trained::Bool
end



function MOVGP(
    X::AbstractArray{T,Nₓ},
    y::AbstractVector{<:AbstractVector},
    kernel::Kernel,
    likelihood::TLikelihood,
    inference::TInference,
    nLatent::Int;
    verbose::Int = 0,
    optimiser = ADAM(0.01),
    atfrequency::Int = 1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean} = ZeroMean(),
    variance::Real = 1.0,
    Aoptimiser = ADAM(0.01),
    ArrayType::UnionAll = Vector,
) where {Nₓ,T<:Real,TLikelihood<:Likelihood,TInference<:Inference}

    @assert length(y) > 0 "y should not be an empty vector"
    nTask = length(y)

    if Nₓ == 2
        X = fill(X, nTask)
    end

    likelihoods = [deepcopy(likelihood) for _ = 1:nTask]
    nf_per_task = zeros(Int64, nTask)
    corrected_y = Vector(undef, nTask)
    for i = 1:nTask
        X[i], corrected_y[i], nf_per_task[i], likelihoods[i] =
            check_data!(X[i], y[i], likelihoods[i])
    end

    @assert inference isa AnalyticVI "The inference object should be of type `AnalyticVI`"
    @assert implemented(likelihood, inference) "The $likelihood is not compatible or implemented with the $inference"

    nSamples = size.(X, 1)
    nDim = size.(X, 2)
    if isa(optimiser, Bool)
        optimiser = optimiser ? ADAM(0.01) : nothing
    end
    if isa(AOptimiser, Bool)
        Aoptimiser = Aoptimiser ? ADAM(0.01) : nothing
    end

    nFeatures = nSamples

    if typeof(mean) <: Real
        mean = ConstantMean(mean)
    elseif typeof(mean) <: AbstractVector{<:Real}
        mean = EmpiricalMean(mean)
    end

    latent_f = ntuple(
        i -> _VGP{T}(
            nFeatures[i],
            kernel,
            mean,
            optimiser,
        ),
        nLatent,
    )

    dpos = Normal(0.5, 0.5)
    dneg = Normal(-0.5, 0.5)
    A = zeros(T, nTask, nf_per_task[1], nLatent)
    for i in eachindex(A)
        p = rand(0:1)
        A[i] = rand(Normal(0.0, 1.0))#p*rand(dpos) + (1-p)*rand(dneg)
    end

    likelihoods .=
        init_likelihood.(
            likelihoods,
            inference,
            nf_per_task,
            nMinibatch,
            nFeatures,
        )
    inference =
        tuple_inference(inference, nLatent, nFeatures, nSamples, nMinibatch)
    inference.xview = view(X, 1:nMinibatch, :)
    inference.yview = view(y, :)

    model = MOVGP{T,TLikelihood,typeof(inference),nTask,nLatent}(
        X,
        corrected_y,
        nSamples,
        nDim,
        nFeatures,
        nLatent,
        nTask,
        nf_per_task,
        latent_f,
        likelihoods,
        inference,
        A,
        Aoptimizer,
        verbose,
        atfrequency,
        false,
    )
    # if isa(inference.optimizer,ALRSVI)
    # init!(model.inference,model)
    # end
    # return model
end

function Base.show(io::IO,model::MOVGP{T,<:Likelihood,<:Inference}) where {T}
    print(io,"Multioutput Variational Gaussian Process with the likelihoods $(model.likelihood) infered by $(model.inference) ")
end

@traitimpl IsMultiOutput{MOVGP}


get_Z(model::MOVGP) = model.X
