mutable struct MCMCIntegrationInference{T<:Real} <: NumericalInference{T}
    ϵ::T #Convergence criteria
    nIter::Integer #Number of steps performed
    optimizer::Optimizer #Learning rate for stochastic updates
    nMC::Int64 #Number of samples for MC Integrations
    Stochastic::Bool #Use of mini-batches
    nSamples::Int64 #Number of samples of the data
    nSamplesUsed::Int64 #Size of mini-batches
    MBIndices::AbstractVector #Indices of the minibatch
    ρ::T #Stochastic Coefficient
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    ∇η₁::AbstractVector{AbstractVector}
    ∇η₂::AbstractVector{AbstractArray}
    ∇μE::AbstractVector{AbstractVector}
    ∇ΣE::AbstractVector{AbstractVector}
    function MCMCIntegrationInference{T}(ϵ::T,nMC::Integer,nIter::Integer,optimizer::Optimizer,Stochastic::Bool,nSamples::Integer,nSamplesUsed::Integer,MBIndices::AbstractVector,ρ::T,flag::Bool) where T
        return new{T}(ϵ,nIter,optimizer,nMC,Stochastic,nSamples,nSamplesUsed,MBIndices,ρ,flag)
    end
    function MCMCIntegrationInference{T}(ϵ::T,nMC::Integer,nIter::Integer,optimizer::Optimizer,Stochastic::Bool,nSamples::Integer,nSamplesUsed::Integer,MBIndices::AbstractVector,ρ::T,flag::Bool,∇μE::AbstractVector{<:AbstractVector},
    ∇ΣE::AbstractVector{<:AbstractVector}) where T
        return new{T}(ϵ,nIter,optimizer,nMC,Stochastic,nSamples,nSamplesUsed,MBIndices,ρ,flag,∇μE,∇ΣE)
    end
end

function defaultn(::Type{MCMCIntegrationInference})
    return 200
end


function compute_grad_expectations(model::VGP{<:Likelihood,<:MCMCIntegrationInference})
    samples = zeros(model.inference.nMC,model.nLatent)
    for i in 1:model.nSample
        samples .= randn(model.inference.nMC,model.nLatent).*[sqrt(model.Σ[k][i,i]) for k in 1:model.nLatent]' .+ [model.μ[k][i] for k in model.nLatent]'
        treat_samples(model,samples,i)
    end
end
