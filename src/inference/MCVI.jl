"""
`MCIntegrationVI(;ϵ::T=1e-5,nMC::Integer=1000,optimiser=Momentum(0.001))`

Variational Inference solver by approximating gradients via MC Integration.

**Keyword arguments**

    - `ϵ::T` : convergence criteria, which can be user defined
    - `nMC::Int` : Number of samples per data point for the integral evaluation
    - `natural::Bool` : Use natural gradients
    - `optimiser` : Optimiser used for the variational updates. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `Momentum(0.01)`
"""
mutable struct MCIntegrationVI{T<:Real,N} <: NumericalVI{T}
    ϵ::T #Convergence criteria
    nIter::Integer #Number of steps performed
    nMC::Int64 #Number of samples for MC Integrations
    Stochastic::Bool #Use of mini-batches
    nSamples::Int64 #Number of samples of the data
    nMinibatch::Int64 #Size of mini-batches
    NaturalGradient::Bool
    ρ::T #Stochastic Coefficient
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    vi_opt::NTuple{N,NVIOptimizer}
    MBIndices::Vector #Indices of the minibatch
    xview::SubArray{T,2,Matrix{T}}
    yview::SubArray
    function MCIntegrationVI{T}(ϵ::T,nMC::Integer,optimiser,Stochastic::Bool,nSamplesUsed::Integer=1,natural::Bool=true) where {T<:Real}
        return new{T,1}(ϵ,0,nMC,Stochastic,1,nSamplesUsed,natural,1.0,true,(NVIOptimizer{T}(0,0,optimiser),))
    end
    function MCIntegrationVI{T,1}(ϵ::T,Stochastic::Bool,nMC::Int,nFeatures::Int,nSamples::Int,nMinibatch::Int,nLatent::Int,optimiser,natural::Bool) where {T}
        vi_opts = ntuple(_->NVIOptimizer{T}(nFeatures,nMinibatch,optimiser),nLatent)
        new{T,nLatent}(ϵ,0,nMC,Stochastic,nSamples,nMinibatch,natural,nSamples/nMinibatch,true,vi_opts,collect(1:nMinibatch))
    end
end


function MCIntegrationVI(;ϵ::T=1e-5,nMC::Integer=1000,optimiser=Momentum(0.01),natural::Bool=true) where {T<:Real}
    MCIntegrationVI{T}(ϵ,nMC,optimiser,false,1,natural)
end

"""
`MCIntegrationSVI(;ϵ::T=1e-5,nMC::Integer=1000,optimiser=Momentum(0.0001))`

Stochastic Variational Inference solver by approximating gradients via Monte Carlo integration

**Argument**

    -`nMinibatch::Integer` : Number of samples per mini-batches

**Keyword arguments**

    - `ϵ::T` : convergence criteria, which can be user defined
    - `nMC::Int` : Number of samples per data point for the integral evaluation
    - `natural::Bool` : Use natural gradients
    - `optimiser` : Optimiser used for the variational updates. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `Momentum()` (ρ=(τ+iter)^-κ)
"""
function MCIntegrationSVI(nMinibatch::Integer;ϵ::T=1e-5,nMC::Integer=200,optimiser=Momentum(0.001),natural::Bool=true) where {T<:Real}
    MCIntegrationVI{T}(ϵ,nMC,optimiser,true,nMinibatch,natural)
end

function tuple_inference(i::TInf,nLatent::Integer,nFeatures::Integer,nSamples::Integer,nMinibatch::Integer) where {TInf <: MCIntegrationVI}
    return TInf(i.ϵ,i.Stochastic,i.nMC,nFeatures,nSamples,nMinibatch,nLatent,i.vi_opt[1].optimiser,i.NaturalGradient)
end

function compute_grad_expectations!(model::AbstractGP{T,L,<:MCIntegrationVI{T,N}}) where {T,L,N}
    raw_samples = randn(model.inference.nMC,model.nLatent)
    samples = similar(raw_samples)
    μ = mean_f(model)
    σ = diag_cov_f(model)
    nSamples = length(model.inference.MBIndices)
    for i in 1:nSamples
        samples .= raw_samples.*[sqrt(σ[k][i]) for k in 1:model.nLatent]' .+ [μ[k][i] for k in 1:N]'
        grad_samples(model,samples,i)
    end
end

function expec_log_likelihood(l::Likelihood,i::MCIntegrationVI{T,N},y,μ,diag_cov) where {T,N}
    raw_samples = randn(i.nMC,N)
    samples = similar(raw_samples)
    nSamples = length(i.MBIndices)
    loglike = 0.0
    for i in 1:nSamples
        samples .= raw_samples.*[sqrt(diag_cov[k][i]) for k in 1:N]' .+ [μ[k][i] for k in 1:N]'
        loglike += mean(mapslices(f->logpdf(l,getindex.(y,i),f),samples,dims=2))
    end
    return loglike
end
