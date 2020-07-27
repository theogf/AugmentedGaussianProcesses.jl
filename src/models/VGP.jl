"""
    VGP(X::AbstractArray{T},y::AbstractVector,
        kernel::Kernel,
        likelihood::LikelihoodType,inference::InferenceType;
        verbose::Int=0,optimiser=ADAM(0.01),atfrequency::Integer=1,
        mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
        IndependentPriors::Bool=true,ArrayType::UnionAll=Vector)

Argument list :

**Mandatory arguments**

 - `X` : input features, should be a matrix N×D where N is the number of observation and D the number of dimension
 - `y` : input labels, can be either a vector of labels for multiclass and single output or a matrix for multi-outputs (note that only one likelihood can be applied)
 - `kernel` : covariance function, a single kernel from the KernelFunctions.jl package
 - `likelihood` : likelihood of the model, currently implemented : Gaussian, Bernoulli (with logistic link), Multiclass (softmax or logistic-softmax) see [`Likelihood Types`](@ref likelihood_user)
 - `inference` : inference for the model, can be analytic, numerical or by sampling, check the model documentation to know what is available for your likelihood see the [`Compatibility Table`](@ref compat_table)

**Keyword arguments**

 - `verbose` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
 - `optimiser` : Optimiser used for the kernel parameters. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `ADAM(0.001)`
 - `atfrequency` : Choose how many variational parameters iterations are between hyperparameters optimization
 - `mean` : PriorMean object, check the documentation on it [`MeanPrior`](@ref meanprior)
 - `IndependentPriors` : Flag for setting independent or shared parameters among latent GPs
 - `ArrayType` : Option for using different type of array for storage (allow for GPU usage)
"""
mutable struct VGP{
    T<:Real,
    TLikelihood<:Likelihood{T},
    TInference<:Inference{T},
    TData<:AbstractDataContainer,
    N,
} <: AbstractGP{T,TLikelihood,TInference,N}
    data::TData # Data container
    f::NTuple{N,VarLatent{T}} # Vector of latent GPs
    likelihood::TLikelihood
    inference::TInference
    verbose::Int #Level of printing information
    atfrequency::Int
    trained::Bool
end


function VGP(
    X::AbstractArray{T},
    y::AbstractVector,
    kernel::Kernel,
    likelihood::TLikelihood,
    inference::TInference;
    verbose::Int = 0,
    optimiser = ADAM(0.01),
    atfrequency::Integer = 1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean} = ZeroMean(),
    ArrayType::UnionAll = Vector,
) where {T<:Real,TLikelihood<:Likelihood,TInference<:Inference}

    X = if X isa AbstractVector
        reshape(X, :, 1)
    else
        X
    end

    y, nLatent, likelihood = check_data!(X, y, likelihood)
    @assert inference isa VariationalInference "The inference object should be of type `VariationalInference` : either `AnalyticVI` or `NumericalVI`"
    @assert !isa(likelihood, GaussianLikelihood) "For a Gaussian Likelihood you should directly use the `GP` model or the `SVGP` model for large datasets"
    @assert implemented(likelihood, inference) "The $likelihood is not compatible or implemented with the $inference"

    nFeatures = nSamples = size(X, 1)
    nDim = size(X, 2)

    if isa(optimiser, Bool)
        optimiser = optimiser ? ADAM(0.01) : nothing
    end

    if typeof(mean) <: Real
        mean = ConstantMean(mean)
    elseif typeof(mean) <: AbstractVector{<:Real}
        mean = EmpiricalMean(mean)
    end

    latentf = ntuple(_ -> _VGP{T}(nFeatures, kernel, mean, optimiser), nLatent)

    likelihood =
        init_likelihood(likelihood, inference, nLatent, nSamples, nFeatures)
    inference =
        tuple_inference(inference, nLatent, nSamples, nSamples, nSamples)
    inference.xview = [view(X, :, :)]
    inference.yview = [view_y(likelihood, y, 1:nSamples)]
    inference.MBIndices = [collect(1:nSamples)]
    VGP{T,TLikelihood,typeof(inference),typeof(data),nLatent}(
        data,
        nFeatures,
        nLatent,
        latentf,
        likelihood,
        inference,
        verbose,
        atfrequency,
        false,
    )
end

function Base.show(io::IO, model::VGP{T,<:Likelihood,<:Inference}) where {T}
    print(
        io,
        "Variational Gaussian Process with a $(model.likelihood) infered by $(model.inference) ",
    )
end


get_Z(m::VGP) = [m.X]
get_Z(m::VGP, i::Int) = m.X
objective(m::VGP) = ELBO(m::VGP)

@traitimpl IsFull{VGP}
