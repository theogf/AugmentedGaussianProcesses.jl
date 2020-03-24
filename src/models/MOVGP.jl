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
    nSamples::Vector{Int64} # Number of data points
    nDim::Vector{Int64} # Number of covariates per data point
    nFeatures::Vector{Int64} # Number of features of the GP (equal to number of points)
    nLatent::Int64 # Number of latent GPs
    nX::Int64
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

    @assert length(y) > 0 "y should not be an empty vector"
    nTask = length(y)

    X = wrap_X_multi(X, nTask)
    nX = length(X)

    if likelihood isa Likelihood
        likelihoods = [deepcopy(likelihood) for _ in 1:nTask]
    end

    nf_per_task = zeros(Int64, nTask)
    corrected_y = Vector(undef, nTask)
    for i in 1:nTask
        corrected_y[i], nf_per_task[i], likelihoods[i] =
            check_data!(X[mod(i, nX) + 1], y[i], likelihoods[i])
    end

    @assert inference isa AnalyticVI "The inference object should be of type `AnalyticVI`"
    @assert all(implemented.(likelihood, Ref(inference))) "One (or all) of the likelihoods:  $likelihoods are not compatible or implemented with $inference"

    nSamples = size.(X, 1)
    nDim = size.(X, 2)
    nFeatures = nSamples
    nMinibatch = nSamples

    if isa(optimiser, Bool)
        optimiser = optimiser ? ADAM(0.01) : nothing
    end

    if isa(Aoptimiser, Bool)
        Aoptimiser = Aoptimiser ? ADAM(0.01) : nothing
    end

    mean = if typeof(mean) <: Real
        ConstantMean(mean)
    elseif typeof(mean) <: AbstractVector{<:Real}
        EmpiricalMean(mean)
    else
        mean
    end

    kernel = if kernel isa Kernel
        [kernel]
    else
        @assert length(kernel) == nLatent "Number of kernels should be equal to the number of tasks"
        kernel
    end
    nKernel = length(kernel)

    latent_f = ntuple(
        i -> _VGP{T}(
            nFeatures[mod(i,nX)+1], #?????
            kernel[mod(i,nKernel)+1],
            mean,
            optimiser,
        ),
        nLatent,
    )

    A = [[randn(nLatent) |> x->x/sqrt(sum(abs2,x)) for i in 1:nf_per_task[j]] for j in 1:nTask]

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
    inference.xview = view.(X, range.(1,nMinibatch,step=1), :)
    inference.yview = view(y, :)

    model = MOVGP{T,TLikelihood,typeof(inference),nTask,nLatent}(
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

function Base.show(io::IO,model::MOVGP{T,<:Likelihood,<:Inference}) where {T}
    print(io,"Multioutput Variational Gaussian Process with the likelihoods $(model.likelihood) infered by $(model.inference) ")
end

@traitimpl IsMultiOutput{MOVGP}
@traitimpl IsFull{MOVGP}

get_Z(m::MOVGP) = m.X
get_Z(m::MOVGP, i::Int) = nX(m) == 1 ? first(m.X) : m.X[i]
objective(m::MOVGP) = ELBO(m)
