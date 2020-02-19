"""
`MCIntegrationVI(;ϵ::T=1e-5,nMC::Integer=1000,optimiser=Momentum(0.001))`

Variational Inference solver by approximating gradients via MC Integration.

**Keyword arguments**

    - `ϵ::T` : convergence criteria, which can be user defined
    - `nMC::Int` : Number of samples per data point for the integral evaluation
    - `optimiser` : Optimiser used for the variational updates. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `Momentum(0.01)`
"""
mutable struct MCIntegrationVI{T<:Real,N} <: NumericalVI{T}
    ϵ::T #Convergence criteria
    nIter::Integer #Number of steps performed
    nMC::Int64 #Number of samples for MC Integrations
    Stochastic::Bool #Use of mini-batches
    nSamples::Int64 #Number of samples of the data
    nMinibatch::Int64 #Size of mini-batches
    ρ::T #Stochastic Coefficient
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    vi_opt::NTuple{N,NVIOptimizer}
    MBIndices::Vector #Indices of the minibatch
    xview::SubArray{T,2,Matrix{T}}
    yview::SubArray
    function MCIntegrationVI{T}(ϵ::T,nMC::Integer,optimiser,Stochastic::Bool,nSamplesUsed::Integer=1) where {T<:Real}
        return new{T,1}(ϵ,0,nMC,Stochastic,1,nSamplesUsed)
    end
end


function MCIntegrationVI(;ϵ::T=1e-5,nMC::Integer=1000,optimiser=Momentum(0.01)) where {T<:Real}
    MCIntegrationVI{T}(ϵ,nMC,optimiser,false,1)
end

"""
`MCIntegrationSVI(;ϵ::T=1e-5,nMC::Integer=1000,optimiser=Momentum(0.0001))`

Stochastic Variational Inference solver by approximating gradients via Monte Carlo integration

**Argument**

    -`nMinibatch::Integer` : Number of samples per mini-batches

**Keyword arguments**

    - `ϵ::T` : convergence criteria, which can be user defined
    - `nMC::Int` : Number of samples per data point for the integral evaluation
    - `optimiser` : Optimiser used for the variational updates. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `Momentum()` (ρ=(τ+iter)^-κ)
"""
function MCIntegrationSVI(nMinibatch::Integer;ϵ::T=1e-5,nMC::Integer=200,optimiser=Momentum(0.001)) where {T<:Real}
    MCIntegrationVI{T}(ϵ,nMC,0,optimiser,true,nMinibatch)
end

function compute_grad_expectations!(model::VGP{T,L,<:MCIntegrationVI}) where {T,L}
    raw_samples = randn(model.inference.nMC,model.nLatent)
    samples = similar(raw_samples)
    for i in 1:model.nSample
        samples .= raw_samples.*[sqrt(model.Σ[k][i,i]) for k in 1:model.nLatent]' .+ [model.μ[k][i] for k in 1:model.nLatent]'
        grad_samples(model,samples,i)
    end
end

function compute_grad_expectations!(model::SVGP{T,L,<:MCIntegrationVI}) where {T,L}
    raw_samples = randn(model.inference.nMC,model.nLatent)
    samples = similar(raw_samples)
    Σ = opt_diag.(model.κ.*model.Σ,model.κ)
    μ = model.κ.*model.μ
    for i in 1:model.nSample
        samples .= raw_samples.*[sqrt(Σ[k][i]) for k in 1:model.nLatent]' .+ [μ[k][i] for k in 1:model.nLatent]'
        grad_samples(model,samples,i)
    end
end

function expec_log_likelihood(l::Likelihood,i::MCIntegrationVI,y,μ,diag_cov) where {T,L}
    raw_samples = randn(i.nMC,i.nLatent)
    samples = similar(raw_samples)
    loglike = 0.0
    for i in 1:model.nSample
        samples .= raw_samples.*[sqrt(diag_cov[k][i]) for k in 1:i.nLatent]' .+ [μ[k][i] for k in 1:i.nLatent]'
        loglike += mean(mapslices(f->logpdf(l,y[i],f),samples,dims=2))
    end
    return loglike
end
