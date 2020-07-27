"""
    VStP(X::AbstractArray{T},y::AbstractArray{T₂,N₂},
        kernel::Kernel,
        likelihood::LikelihoodType,inference::InferenceType,ν::T₃;
        verbose::Int=0,optimiser=ADAM(0.01),atfrequency::Integer=1,
        mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
        IndependentPriors::Bool=true,ArrayType::UnionAll=Vector)

Class for variational Student-T Processes models (non-sparse)
Argument list :

**Mandatory arguments**

 - `X` : input features, should be a matrix N×D where N is the number of observation and D the number of dimension
 - `y` : input labels, can be either a vector of labels for multiclass and single output or a matrix for multi-outputs (note that only one likelihood can be applied)
 - `kernel` : covariance function, can be either a single kernel or a collection of kernels for multiclass and multi-outputs models
 - `likelihood` : likelihood of the model, currently implemented : Gaussian, Bernoulli (with logistic link), Multiclass (softmax or logistic-softmax) see [`Likelihood Types`](@ref likelihood_user)
 - `inference` : inference for the model, can be analytic, numerical or by sampling, check the model documentation to know what is available for your likelihood see the [`Compatibility Table`](@ref compat_table)
 - `ν` : Number of degrees of freedom

**Keyword arguments**

 - `verbose` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
 - `optimiser` : Optimiser used for the kernel parameters. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `ADAM(0.001)`
 - `atfrequency` : Choose how many variational parameters iterations are between hyperparameters optimization
 - `mean` : PriorMean object, check the documentation on it [`MeanPrior`](@ref meanprior)
 - `IndependentPriors` : Flag for setting independent or shared parameters among latent GPs
 - `ArrayType` : Option for using different type of array for storage (allow for GPU usage)
"""
mutable struct VStP{
    T<:Real,
    TLikelihood<:Likelihood{T},
    TInference<:Inference{T},
    TData<:AbstractDataContainer,
    N,
} <: AbstractGP{T,TLikelihood,TInference,N}
    data::TData
    ν::T # Number of degrees of freedom
    nLatent::Int64 # Number pf latent GPs
    f::NTuple{N,TVarLatent{T}}
    likelihood::TLikelihood
    inference::TInference
    verbose::Int64 #Level of printing information
    atfrequency::Int64
    Trained::Bool
end


function VStP(
    X::AbstractArray{T},
    y::AbstractVector,
    kernel::Kernel,
    likelihood::TLikelihood,
    inference::TInference,
    ν::Real;
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
    @assert implemented(likelihood, inference) "The $likelihood is not compatible or implemented with the $inference"

    @assert ν > 1 "ν should be bigger than 1"
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

    latentf =
        ntuple(_ -> _VStP{T}(ν, nFeatures, kernel, mean, optimiser), nLatent)

    likelihood =
        init_likelihood(likelihood, inference, nLatent, nSamples, nFeatures)
    inference =
        tuple_inference(inference, nLatent, nSamples, nSamples, nSamples)
    inference.xview = [view(X, :, :)]
    inference.yview = [view_y(likelihood, y, 1:nSamples)]
    inference.MBIndices = [collect(1:nSamples)]
    VStP{T,TLikelihood,typeof(inference),nLatent}(
        X,
        y,
        ν,
        nFeatures,
        nDim,
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

function Base.show(io::IO, model::VStP)
    print(
        io,
        "Variational Student-T Process with a $(model.likelihood) infered by $(model.inference) ",
    )
end

function local_prior_updates!(model::VStP, X)
    for gp in model.f
        local_prior_updates!(gp, X)
    end
end

function local_prior_updates!(gp::TVarLatent, X)
    gp.l² = 0.5 * ( gp.ν + digp.dim + invquad(gp.K, gp.μ - gp.μ₀(X)) + opt_trace(inv(gp.K).mat, gp.Σ))
    gp.χ = (gp.ν + gp.dim) / (gp.ν .+ gp.l²)
end

get_X(m::VStP) = m.X
get_Z(m::VStP) = [m.X]
get_Z(m::VStP, i::Int) = m.X
objective(m::VStP) = ELBO(m)

@traitimpl IsFull{VStP}
