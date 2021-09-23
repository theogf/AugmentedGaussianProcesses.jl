mutable struct MCIntegrationVI{T<:Real} <: NumericalVI{T}
    nMC::Int64 #Number of samples for MC Integrations
    clipping::T
    ϵ::T #Convergence criteria
    n_iter::Integer #Number of steps performed
    stoch::Bool # Flag for stochastic optimization
    batchsize::Int #Size of mini-batches
    ρ::T # Scaling coeff. for stoch. opt.
    NaturalGradient::Bool
    HyperParametersUpdated::Bool # Flag for updating kernel matrix
    vi_opt::NVIOptimizer
    function MCIntegrationVI{T}(
        ϵ::T,
        nMC::Int,
        optimiser,
        Stochastic::Bool,
        clipping::Real,
        batchsize::Int,
        natural::Bool,
    ) where {T<:Real}
        return new{T}(
            nMC,
            clipping,
            ϵ,
            0,
            Stochastic,
            batchsize,
            one(T),
            natural,
            true,
            NVIOptimizer(optimiser),
        )
    end
end

"""
    MCIntegrationVI(;ϵ::T=1e-5, nMC::Integer=1000, clipping::Real=Inf, natural::Bool=true, optimiser=Momentum(0.001))

Variational Inference solver by approximating gradients via MC Integration.
It means the expectation `E[log p(y|f)]` as well as its gradients is computed
by sampling from q(f).

## Keyword arguments
- `ϵ::Real` : convergence criteria, which can be user defined
- `nMC::Int` : Number of samples per data point for the integral evaluation
- `clipping::Real` : Limit the gradients values to avoid overshooting
- `natural::Bool` : Use natural gradients
- `optimiser` : Optimiser used for the variational updates. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `Momentum(0.01)`
"""
MCIntegrationVI

function MCIntegrationVI(;
    ϵ::T=1e-5,
    nMC::Integer=1000,
    optimiser=Momentum(0.01),
    clipping::Real=Inf,
    natural::Bool=true,
) where {T<:Real}
    return MCIntegrationVI{T}(ϵ, nMC, optimiser, false, clipping, 1, natural)
end

"""
    MCIntegrationSVI(batchsize::Int; ϵ::Real=1e-5, nMC::Integer=1000, clipping=Inf, natural=true, optimiser=Momentum(0.0001))

Stochastic Variational Inference solver by approximating gradients via Monte Carlo integration when using minibatches
See [`MCIntegrationVI`](@ref) for more explanations.

## Argument

-`batchsize::Integer` : Number of samples per mini-batches

## Keyword arguments

- `ϵ::T` : convergence criteria, which can be user defined
- `nMC::Int` : Number of samples per data point for the integral evaluation
- `clipping::Real` : Limit the gradients values to avoid overshooting
- `natural::Bool` : Use natural gradients
- `optimiser` : Optimiser used for the variational updates. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `Momentum()` (ρ=(τ+iter)^-κ)
"""
MCIntegrationSVI

function MCIntegrationSVI(
    batchsize::Integer;
    ϵ::T=1e-5,
    nMC::Integer=200,
    optimiser=Momentum(0.001),
    clipping::Real=0.0,
    natural::Bool=true,
) where {T<:Real}
    return MCIntegrationVI{T}(ϵ, nMC, optimiser, true, clipping, batchsize, natural)
end

function grad_expectations!(
    m::AbstractGPModel{T,L,<:MCIntegrationVI{T}}, state, y
) where {T,L}
    raw_samples = randn(T, inference(m).nMC, n_latent(m))
    samples = similar(raw_samples)
    μ = mean_f(m, state.kernel_matrices)
    σ² = var_f(m, state.kernel_matrices)
    num_sample = batchsize(m)
    opt_state = state.opt_state
    for j in 1:num_sample # Loop over every data point
        samples .=
            raw_samples .* sqrt.([σ²[k][j] for k in 1:n_latent(m)])' .+ [μ[k][j] for k in 1:n_latent(m)]'
        grad_samples!(m, samples, opt_state, y[j, :], j) # Compute the gradient for data point j
    end
end

function expec_loglikelihood(
    l::AbstractLikelihood, i::MCIntegrationVI{T}, y, μ, σ²
) where {T} # μ and σ² are tuples of vectors
    num_latent = n_latent(l)
    raw_samples = randn(T, i.nMC, num_latent) # dimension nMC x nLatent
    # samples = similar(raw_samples)
    num_sample = batchsize(i)
    tot = 0.0
    for j in 1:num_sample # Loop over every data point
        samples =
            raw_samples .* sqrt.([σ²[k][j] for k in 1:num_latent])' .+ [μ[k][j] for k in 1:num_latent]'
        # samples is of dimension nMC x nLatent again
        y_j = y[j, :] # Obtain the label for data point j
        for f in eachrow(samples)
            # We now compute the loglikelihood over every sample
            tot += loglikelihood(l, y_j, f)
        end
    end
    return tot / i.nMC
end
