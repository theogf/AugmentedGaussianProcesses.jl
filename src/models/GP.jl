"""
    GP(args...; kwargs...)

Gaussian Process

## Arguments

 - `X` : input features, should be a matrix N×D where N is the number of observation and D the number of dimension
 - `y` : input labels, can be either a vector of labels for multiclass and single output or a matrix for multi-outputs (note that only one likelihood can be applied)
 - `kernel` : covariance function, can be either a single kernel or a collection of kernels for multiclass and multi-outputs models

## Keyword arguments
- `noise` : Variance of the likelihood
- `opt_noise` : Flag for optimizing the variance by using the formul σ=Σ(y-f)^2/N
- `mean` : Option for putting a prior mean
- `verbose` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
- `optimiser` : Optimiser used for the kernel parameters. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `ADAM(0.001)`
- `IndependentPriors` : Flag for setting independent or shared parameters among latent GPs
- `atfrequency` : Choose how many variational parameters iterations are between hyperparameters optimization
- `mean` : PriorMean object, check the documentation on it [`MeanPrior`](@ref meanprior)
"""
mutable struct GP{
    T<:Real,
    TLikelihood<:AbstractLikelihood,
    TInference<:AbstractInference,
    TData<:DataContainer,
} <: AbstractGPModel{T,TLikelihood,TInference,1}
    data::TData
    f::LatentGP{T}
    likelihood::TLikelihood
    inference::TInference
    verbose::Int #Level of printing information
    atfrequency::Int
    trained::Bool
end

function GP(
    X::AbstractArray,
    y::AbstractArray,
    kernel::Kernel;
    noise::Real=1e-5,
    opt_noise=true,
    verbose::Int=0,
    optimiser=ADAM(0.01),
    atfrequency::Int=1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
    obsdim=1,
)
    X, T = wrap_X(X, obsdim)
    likelihood = GaussianLikelihood(noise; opt_noise=opt_noise)
    inference = Analytic()

    y = check_data!(y, likelihood)
    data = wrap_data(X, y)
    n_feature = n_sample(data)
    if isa(optimiser, Bool)
        optimiser = optimiser ? ADAM(0.01) : nothing
    end

    latentf = LatentGP(T, n_feature, kernel, mean, optimiser)

    model = GP(data, latentf, likelihood, inference, verbose, atfrequency, false)
    model = train!(model, 1)
    return model
end

function Base.show(io::IO, model::GP)
    return print(
        io, "Gaussian Process with a $(likelihood(model)) infered by $(inference(model)) "
    )
end

n_latent(::GP) = 1
Zviews(model::GP) = [input(model)]

@traitimpl IsFull{GP}

### Special case where the ELBO is equal to the marginal likelihood

function post_step!(m::GP)
    f = m.f
    l = likelihood(m)
    f.post.Σ = pr_cov(f) + first(l.σ²) * I
    return f.post.α .= cov(f) \ (yview(m) - pr_mean(f, xview(m)))
end

objective(m::GP) = log_py(m)

function log_py(m::GP)
    f = m.f
    return -0.5 *
           (dot(yview(m), cov(f) \ yview(m)) + logdet(cov(f)) + nFeatures(m) * log(twoπ))
end
