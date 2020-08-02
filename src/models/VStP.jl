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
    f::NTuple{N,TVarLatent{T}}
    likelihood::TLikelihood
    inference::TInference
    verbose::Int64 #Level of printing information
    atfrequency::Int64
    trained::Bool
end


function VStP(
    X::AbstractArray{<:Real},
    y::AbstractVector,
    kernel::Kernel,
    likelihood::TLikelihood,
    inference::Inference,
    ν::Real;
    verbose::Int = 0,
    optimiser = ADAM(0.01),
    atfrequency::Int = 1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean} = ZeroMean(),
    obsdim::Int = 1
) where {TLikelihood<:Likelihood}

    X, T = wrap_X(X, obsdim)
    y, nLatent, likelihood = check_data!(y, likelihood)

    @assert inference isa VariationalInference "The inference object should be of type `VariationalInference` : either `AnalyticVI` or `NumericalVI`"
    @assert implemented(likelihood, inference) "The $likelihood is not compatible or implemented with the $inference"

    data = wrap_data(X, y)

    ν > 1  || error("ν should be bigger than 1")

    nFeatures = nSamples(data)

    if isa(optimiser, Bool)
        optimiser = optimiser ? ADAM(0.01) : nothing
    end

    if typeof(mean) <: Real
        mean = ConstantMean(mean)
    elseif typeof(mean) <: AbstractVector{<:Real}
        mean = EmpiricalMean(mean)
    end

    latentf =
        ntuple(_ -> TVarLatent(T, ν, nFeatures, kernel, mean, optimiser), nLatent)

    likelihood =
        init_likelihood(likelihood, inference, nLatent, nSamples(data), nFeatures)
    xview = view_x(data, 1:nSamples(data))
    yview = view_y(likelihood, data, 1:nSamples(data))
    inference =
        tuple_inference(inference, nLatent, nSamples(data), nSamples(data), nSamples(data), xview, yview)
    VStP{T,TLikelihood,typeof(inference),typeof(data),nLatent}(
        data,
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
        "Variational Student-T Process with a $(likelihood(model)) infered by $(inference(model)) ",
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

Zviews(m::VStP) = [input(m)]
objective(m::VStP) = ELBO(m)

@traitimpl IsFull{VStP}
