mutable struct MCIntegrationVI{T<:Real,N,Tx,Ty} <: NumericalVI{T}
    nMC::Int64 #Number of samples for MC Integrations
    clipping::T
    ϵ::T #Convergence criteria
    nIter::Integer #Number of steps performed
    stoch::Bool # Flag for stochastic optimization
    nSamples::Int #Number of samples of the data
    nMinibatch::Int #Size of mini-batches
    ρ::T # Scaling coeff. for stoch. opt.
    NaturalGradient::Bool
    HyperParametersUpdated::Bool # Flag for updating kernel matrix
    vi_opt::NTuple{N,NVIOptimizer}
    MBIndices::Vector{Int} #Indices of the minibatch
    xview::Tx
    yview::Ty
    function MCIntegrationVI{T}(
        ϵ::T,
        nMC::Int,
        optimiser,
        Stochastic::Bool,
        clipping::Real,
        nMinibatch::Int,
        natural::Bool,
    ) where {T<:Real}
        return new{T,1,Vector{T},Vector{T}}(
            nMC,
            clipping,
            ϵ,
            0,
            Stochastic,
            0,
            nMinibatch,
            one(T),
            natural,
            true,
            (NVIOptimizer{T}(0, 0, optimiser),),
        )
    end
    function MCIntegrationVI{T}(
        ϵ::Real,
        Stochastic::Bool,
        nMC::Int,
        clipping::Real,
        nFeatures::Vector{<:Int},
        nSamples::Int,
        nMinibatch::Int,
        nLatent::Int,
        optimiser,
        natural::Bool,
        xview::Tx,
        yview::Ty,
    ) where {T,Tx,Ty}
        vi_opts = ntuple(i -> NVIOptimizer{T}(nFeatures[i], nMinibatch, optimiser), nLatent)
        return new{T,nLatent,Tx,Ty}(
            nMC,
            clipping,
            ϵ,
            0,
            Stochastic,
            nSamples,
            nMinibatch,
            T(nSamples / nMinibatch),
            natural,
            true,
            vi_opts,
            1:nMinibatch,
            xview,
            yview,
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
    MCIntegrationSVI(nMinibatch::Int; ϵ::Real=1e-5, nMC::Integer=1000, clipping=Inf, natural=true, optimiser=Momentum(0.0001))

Stochastic Variational Inference solver by approximating gradients via Monte Carlo integration when using minibatches
See [`MCIntegrationVI`](@ref) for more explanations.

## Argument

-`nMinibatch::Integer` : Number of samples per mini-batches

## Keyword arguments

- `ϵ::T` : convergence criteria, which can be user defined
- `nMC::Int` : Number of samples per data point for the integral evaluation
- `clipping::Real` : Limit the gradients values to avoid overshooting
- `natural::Bool` : Use natural gradients
- `optimiser` : Optimiser used for the variational updates. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `Momentum()` (ρ=(τ+iter)^-κ)
"""
MCIntegrationSVI

function MCIntegrationSVI(
    nMinibatch::Integer;
    ϵ::T=1e-5,
    nMC::Integer=200,
    optimiser=Momentum(0.001),
    clipping::Real=0.0,
    natural::Bool=true,
) where {T<:Real}
    return MCIntegrationVI{T}(ϵ, nMC, optimiser, true, clipping, nMinibatch, natural)
end

function tuple_inference(
    i::MCIntegrationVI{T},
    nLatent::Int,
    nFeatures::Vector{<:Int},
    nSamples::Int,
    nMinibatch::Int,
    xview::Tx,
    yview::Ty,
) where {T,Tx,Ty}
    return MCIntegrationVI{T}(
        conv_crit(i),
        isStochastic(i),
        i.nMC,
        i.clipping,
        nFeatures,
        nSamples,
        nMinibatch,
        nLatent,
        i.vi_opt[1].optimiser,
        i.NaturalGradient,
        xview,
        yview,
    )
end

function grad_expectations!(m::AbstractGP{T,L,<:MCIntegrationVI{T,N}}) where {T,L,N}
    raw_samples = randn(T, inference(m).nMC, nLatent(m))
    samples = similar(raw_samples)
    μ = mean_f(m)
    σ² = var_f(m)
    nSamples = length(MBIndices(m))
    for j in 1:nSamples # Loop over every data point
        samples .= raw_samples .* sqrt.(getindex.(σ², j))' .+ getindex.(μ, j)'
        grad_samples(m, samples, j) # Compute the gradient for data point j
    end
end

function expec_loglikelihood(
    l::AbstractLikelihood, i::MCIntegrationVI{T,N}, y, μ_f, var_f
) where {T,N}
    raw_samples = randn(T, inference(m).nMC, nLatent(m)) # dimension nMC x nLatent
    # samples = similar(raw_samples)
    nSamples = length(MBIndices(i))
    return sum(1:nSamples) do j # Loop over every data point
        samples = raw_samples .* sqrt.(getindex.(var_f, j))' .+ getindex.(μ_f, j)'
        # samples is of dimension nMC x nLatent again
        y_j = getindex.(y, j) # Obtain the label for data point j
        return sum(eachrow(samples)) do f # We now compute the loglikelihood over every sample
            return loglikelihood(l, y_j, f)
        end
    end / i.nMC
end