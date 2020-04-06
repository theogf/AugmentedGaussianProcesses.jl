"""
    MCIntegrationVI(;ϵ::T=1e-5,nMC::Integer=1000,optimiser=Momentum(0.001))

Variational Inference solver by approximating gradients via MC Integration.

**Keyword arguments**

    - `ϵ::T` : convergence criteria, which can be user defined
    - `nMC::Int` : Number of samples per data point for the integral evaluation
    - `natural::Bool` : Use natural gradients
    - `optimiser` : Optimiser used for the variational updates. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `Momentum(0.01)`
"""
mutable struct MCIntegrationVI{T<:Real,N} <: NumericalVI{T}
    nMC::Int64 #Number of samples for MC Integrations
    clipping::T
    ϵ::T #Convergence criteria
    nIter::Integer #Number of steps performed
    Stochastic::Bool #Use of mini-batches
    nSamples::Vector{Int64} #Number of samples of the data
    nMinibatch::Vector{Int64} #Size of mini-batches
    ρ::Vector{T} #Stochastic Coefficient
    NaturalGradient::Bool
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    vi_opt::NTuple{N,NVIOptimizer}
    MBIndices::Vector{Vector{Int64}} #Indices of the minibatch
    xview::Vector
    yview::Vector
    function MCIntegrationVI{T}(
        ϵ::T,
        nMC::Integer,
        optimiser,
        Stochastic::Bool,
        clipping::Real,
        nMinibatch::Integer,
        natural::Bool,
    ) where {T<:Real}
        return new{T,1}(
            nMC,
            clipping,
            ϵ,
            0,
            Stochastic,
            [0],
            [nMinibatch],
            ones(T, 1),
            natural,
            true,
            (NVIOptimizer{T}(0, 0, optimiser),),
        )
    end
    function MCIntegrationVI{T,1}(
        ϵ::T,
        Stochastic::Bool,
        nMC::Int,
        clipping::Real,
        nFeatures::Vector{<:Int},
        nSamples::Vector{<:Int},
        nMinibatch::Vector{<:Int},
        nLatent::Int,
        optimiser,
        natural::Bool,
    ) where {T}
        vi_opts = ntuple(
            i -> NVIOptimizer{T}(nFeatures[i], nMinibatch[i], optimiser),
            nLatent,
        )
        new{T,nLatent}(
            nMC,
            clipping,
            ϵ,
            0,
            Stochastic,
            nSamples,
            nMinibatch,
            T.(nSamples ./ nMinibatch),
            natural,
            true,
            vi_opts,
            range.(1, nMinibatch, step = 1),
        )
    end
end


function MCIntegrationVI(;
    ϵ::T = 1e-5,
    nMC::Integer = 1000,
    optimiser = Momentum(0.01),
    clipping::Real = 0.0,
    natural::Bool = true,
) where {T<:Real}
    MCIntegrationVI{T}(ϵ, nMC, optimiser, false, clipping, 1, natural)
end

"""
    MCIntegrationSVI(;ϵ::T=1e-5,nMC::Integer=1000,optimiser=Momentum(0.0001))

Stochastic Variational Inference solver by approximating gradients via Monte Carlo integration

**Argument**

    -`nMinibatch::Integer` : Number of samples per mini-batches

**Keyword arguments**

    - `ϵ::T` : convergence criteria, which can be user defined
    - `nMC::Int` : Number of samples per data point for the integral evaluation
    - `natural::Bool` : Use natural gradients
    - `optimiser` : Optimiser used for the variational updates. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `Momentum()` (ρ=(τ+iter)^-κ)
"""
function MCIntegrationSVI(
    nMinibatch::Integer;
    ϵ::T = 1e-5,
    nMC::Integer = 200,
    optimiser = Momentum(0.001),
    clipping::Real = 0.0,
    natural::Bool = true,
) where {T<:Real}
    MCIntegrationVI{T}(ϵ, nMC, optimiser, true, clipping, nMinibatch, natural)
end

function tuple_inference(
    i::TInf,
    nLatent::Int,
    nFeatures::Vector{<:Int},
    nSamples::Vector{<:Int},
    nMinibatch::Vector{<:Int},
) where {TInf<:MCIntegrationVI}
    return TInf(
        i.ϵ,
        i.Stochastic,
        i.nMC,
        i.clipping,
        nFeatures,
        nSamples,
        nMinibatch,
        nLatent,
        i.vi_opt[1].optimiser,
        i.NaturalGradient,
    )
end

function grad_expectations!(
    model::AbstractGP{T,L,<:MCIntegrationVI{T,N}},
) where {T,L,N}
    raw_samples = randn(model.inference.nMC, model.nLatent)
    samples = similar(raw_samples)
    μ = mean_f(model)
    σ = diag_cov_f(model)
    nSamples = length(model.inference.MBIndices)
    for i = 1:nSamples
        samples .=
            raw_samples .* [sqrt(σ[k][i]) for k = 1:model.nLatent]' .+ [μ[k][i] for k = 1:N]'
        grad_samples(model, samples, i)
    end
end

function expec_log_likelihood(
    l::Likelihood,
    i::MCIntegrationVI{T,N},
    y,
    μ,
    diag_cov,
) where {T,N}
    raw_samples = randn(i.nMC, N)
    samples = similar(raw_samples)
    nSamples = length(i.MBIndices)
    loglike = 0.0
    for i = 1:nSamples
        samples .=
            raw_samples .* [sqrt(diag_cov[k][i]) for k = 1:N]' .+ [μ[k][i] for k = 1:N]'
        loglike += mean(mapslices(
            f -> logpdf(l, getindex.(y, i), f),
            samples,
            dims = 2,
        ))
    end
    return loglike
end
