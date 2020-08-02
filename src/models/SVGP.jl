"""
Class for sparse variational Gaussian Processes


    SVGP(X::AbstractArray{T1},y::AbstractVector{T2},kernel::Kernel,
        likelihood::LikelihoodType,inference::InferenceType, nInducingPoints::Int;
        verbose::Int=0,optimiser=ADAM(0.001),atfrequency::Int=1,
        mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
        Zoptimiser=false,
        ArrayType::UnionAll=Vector)

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
- `Zoptimiser` : Optimiser used for the inducing points locations. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `ADAM(0.001)`
 - `ArrayType` : Option for using different type of array for storage (allow for GPU usage)
"""
mutable struct SVGP{
    T<:Real,
    TLikelihood<:Likelihood{T},
    TInference<:Inference,
    TData<:AbstractDataContainer,
    N,
} <: AbstractGP{T,TLikelihood,TInference,N}
    data::TData
    nFeatures::Vector{Int} # Number of features of each latent
    f::NTuple{N,SparseVarLatent{T}}
    likelihood::TLikelihood
    inference::TInference
    verbose::Int64
    atfrequency::Int64
    trained::Bool
end

function SVGP(
    X::AbstractArray{<:Real},
    y::AbstractVector,
    kernel::Kernel,
    likelihood::Likelihood,
    inference::Inference,
    nInducingPoints::Int;
    verbose::Int = 0,
    optimiser = ADAM(0.01),
    atfrequency::Int = 1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean} = ZeroMean(),
    Zoptimiser = false,
    obsdim::Int = 1
) where {TLikelihood<:Likelihood}
    SVGP(
        X,
        y,
        kernel,
        likelihood,
        inference,
        KmeansIP(X, nInducingPoints),
        verbose = verbose,
        optimiser = optimiser,
        atfrequency = atfrequency,
        mean = mean,
        obsdim = obsdim
    )
end

function SVGP(
    X::AbstractArray{<:Real},
    y::AbstractVector,
    kernel::Kernel,
    likelihood::TLikelihood,
    inference::Inference,
    nInducingPoints::AbstractInducingPoints;
    verbose::Int = 0,
    optimiser = ADAM(0.01),
    atfrequency::Int = 1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean} = ZeroMean(),
    Zoptimiser = false,
    obsdim::Int = 1,
) where {TLikelihood<:Likelihood}

    X, T = wrap_X(X, obsdim)
    y, nLatent, likelihood = check_data!(y, likelihood)

    @assert inference isa VariationalInference "The inference object should be of type `VariationalInference` : either `AnalyticVI` or `NumericalVI`"
    @assert implemented(likelihood, inference) "The $likelihood is not compatible or implemented with the $inference"

    data = wrap_data(X, y)
    if isa(optimiser, Bool)
        optimiser = optimiser ? ADAM(0.001) : nothing
    end

    Z = init_Z(nInducingPoints, Zoptimiser)

    nFeatures = length(Z)

    if typeof(mean) <: Real
        mean = ConstantMean(mean)
    elseif typeof(mean) <: AbstractVector{<:Real}
        mean = EmpiricalMean(mean)
    end

    S = if isStochastic(inference)
        @assert 0 < nMinibatch(inference) < nSamples(data) "The size of mini-batch $(nMinibatch(inference)) is incorrect (negative or bigger than number of samples), please set nMinibatch correctly in the inference object"
        nMinibatch(inference)
    else
        nSamples(data)
    end

    latentf = ntuple(
        _ -> SparseVarLatent(T, nFeatures, S, Z, kernel, mean, optimiser),
        nLatent,
    )

    likelihood =
        init_likelihood(likelihood, inference, nLatent, S, nFeatures)
    xview = view_x(data, collect(1:S))
    yview = view_y(likelihood, data, collect(1:S))
    inference =
        tuple_inference(inference, nLatent, nFeatures, nSamples(data), S, xview, yview)

    model = SVGP{T,TLikelihood,typeof(inference),typeof(data),nLatent}(
        data,
        fill(nFeatures, nLatent), # WARNING workaround
        latentf,
        likelihood,
        inference,
        verbose,
        atfrequency,
        false,
    )
    if isa(optimiser, ALRSVI)
        init!(model)
    end
    return model
end

function Base.show(io::IO, model::SVGP{T,<:Likelihood,<:Inference}) where {T}
    print(
        io,
        "Sparse Variational Gaussian Process with a $(model.likelihood) infered by $(model.inference) ",
    )
end

Zviews(m::SVGP) = Zview.(m.f)
objective(m::SVGP) = ELBO(m)

@traitimpl IsSparse{SVGP}
