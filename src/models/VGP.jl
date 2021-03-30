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
    TLikelihood<:AbstractLikelihood,
    TInference<:AbstractInference,
    TData<:AbstractDataContainer,
    N,
} <: AbstractGP{T,TLikelihood,TInference,N}
    data::TData # Data container
     nFeatures::Vector{Int}
    f::NTuple{N,VarLatent{T}} # Vector of latent GPs
    likelihood::TLikelihood
    inference::TInference
    verbose::Int #Level of printing information
    atfrequency::Int
    trained::Bool
end


function VGP(
    X::AbstractArray,
    y::AbstractVector,
    kernel::Kernel,
    likelihood::AbstractLikelihood,
    inference::AbstractInference;
    verbose::Int = 0,
    optimiser = ADAM(0.01),
    atfrequency::Integer = 1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean} = ZeroMean(),
    obsdim::Int = 1,
)

    X, T = wrap_X(X, obsdim)
    y, nLatent, likelihood = check_data!(y, likelihood)

    inference isa VariationalInference || error("The inference object should be of type `VariationalInference` : either `AnalyticVI` or `NumericalVI`")
    !isa(likelihood, GaussianLikelihood) || error("For a Gaussian Likelihood you should directly use the `GP` model or the `SVGP` model for large datasets")
    implemented(likelihood, inference) ||  error("The $likelihood is not compatible or implemented with the $inference")

    data = wrap_data(X, y)

    if isa(optimiser, Bool)
        optimiser = optimiser ? ADAM(0.01) : nothing
    end

    nFeatures = nSamples(data)

    if typeof(mean) <: Real
        mean = ConstantMean(mean)
    elseif typeof(mean) <: AbstractVector{<:Real}
        mean = EmpiricalMean(mean)
    end

    latentf = ntuple(_ -> VarLatent(T, nFeatures, kernel, mean, optimiser), nLatent)

    likelihood =
        init_likelihood(likelihood, inference, nLatent, nSamples(data))
    xview = view_x(data, :)
    yview = view_y(likelihood, data, 1:nSamples(data))
    inference =
        tuple_inference(inference, nLatent, nFeatures, nSamples(data), nSamples(data), xview, yview)
    return VGP{T,typeof(likelihood),typeof(inference),typeof(data),nLatent}(
        data,
        fill(nFeatures, nLatent),
        latentf,
        likelihood,
        inference,
        verbose,
        atfrequency,
        false,
    )
end

function Base.show(io::IO, model::VGP)
    print(
        io,
        "Variational Gaussian Process with a $(likelihood(model)) infered by $(inference(model)) ",
    )
end


Zviews(m::VGP) = [input(m)]
objective(m::VGP) = ELBO(m::VGP)

@traitimpl IsFull{VGP}
