mutable struct MCIntegrationVI{T<:Real} <: NumericalVI{T}
    ϵ::T #Convergence criteria
    nIter::Integer #Number of steps performed
    optimizer::LatentArray{Optimizer} #Learning rate for stochastic updates
    nMC::Int64 #Number of samples for MC Integrations
    Stochastic::Bool #Use of mini-batches
    nSamples::Int64 #Number of samples of the data
    nSamplesUsed::Int64 #Size of mini-batches
    MBIndices::AbstractVector #Indices of the minibatch
    ρ::T #Stochastic Coefficient
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    ∇η₁::AbstractVector{AbstractVector}
    ∇η₂::AbstractVector{AbstractArray}
    ν::AbstractVector{AbstractVector}
    λ::AbstractVector{AbstractVector}
    x::SubArray{T,2,Matrix{T}}#,Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}}},true}
    y::LatentArray{SubArray}
    function MCIntegrationVI{T}(ϵ::T,nMC::Integer,nIter::Integer,optimizer::Opt,Stochastic::Bool,nSamplesUsed::Integer=1) where {T<:Real,Opt<:Optimizer}
        return new{T}(ϵ,nIter,[optimizer],[optimizer],nMC,Stochastic,1,nSamplesUsed)
    end
end

""" `MCIntegrationVI(;ϵ::T=1e-5,nMC::Integer=1000,optimizer::Optimizer=Adam(α=0.1))`

Constructor for Variational Inference via MC Integration approximation.

**Keyword arguments**

    - `ϵ::T` : convergence criteria, which can be user defined
    - `nMC::Int` : Number of samples per data point for the integral evaluation
    - `optimizer::Optimizer` : Optimizer used for the variational updates. Should be an Optimizer object from the [GradDescent.jl]() package. Default is `Adam()`
"""
function MCIntegrationVI(;ϵ::T=1e-5,nMC::Integer=1000,optimizer::Optimizer=Momentum(η=0.01)) where {T<:Real}
    MCIntegrationVI{T}(ϵ,nMC,0,optimizer,false,1)
end

""" `MCIntegrationSVI(;ϵ::T=1e-5,nMC::Integer=1000,optimizer::Optimizer=Adam(α=0.1))`

Constructor for Stochastic Variational Inference via MC integration approximation.

**Argument**

    -`nMinibatch::Integer` : Number of samples per mini-batches

**Keyword arguments**

    - `ϵ::T` : convergence criteria, which can be user defined
    - `nMC::Int` : Number of samples per data point for the integral evaluation
    - `optimizer::Optimizer` : Optimizer used for the variational updates. Should be an Optimizer object from the [GradDescent.jl]() package. Default is `Adam()`
"""
function MCIntegrationSVI(nMinibatch::Integer;ϵ::T=1e-5,nMC::Integer=200,optimizer::Optimizer=Momentum(η=0.001)) where {T<:Real}
    MCIntegrationVI{T}(ϵ,nMC,0,optimizer,true,nMinibatch)
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

function compute_log_expectations(model::VGP{T,L,<:MCIntegrationVI}) where {T,L}
    raw_samples = randn(model.inference.nMC,model.nLatent)
    samples = similar(raw_samples)
    loglike = 0.0
    for i in 1:model.nSample
        samples .= raw_samples.*[sqrt(model.Σ[k][i,i]) for k in 1:model.nLatent]' .+ [model.μ[k][i] for k in 1:model.nLatent]'
        loglike += mean(mapslices(f->log(pdf(model.likelihood,model.likelihood.y_class[i],f)),samples,dims=2))
    end
    return loglike
end


function compute_log_expectations(model::SVGP{T,L,<:MCIntegrationVI}) where {T,L}
    raw_samples = randn(model.inference.nMC,model.nLatent)
    samples = similar(raw_samples)
    loglike = 0.0
    Σ = opt_diag.(model.κ.*model.Σ,model.κ)
    μ = model.κ.*model.μ
    for i in model.inference.MBIndices
        samples .= raw_samples.*[sqrt(Σ[k][i]) for k in 1:model.nLatent]' .+ [μ[k][i] for k in 1:model.nLatent]'
        loglike += mean(mapslices(f->log(pdf(model.likelihood,model.likelihood.y_class[i],f)),samples,dims=2))
    end
    return model.inference.ρ*loglike
end
