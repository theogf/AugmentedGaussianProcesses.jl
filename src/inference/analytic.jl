""" Solve conjugate or conditionally conjugate likelihoods (especially valid for augmented likelihoods) """
mutable struct AnalyticInference{T<:Real} <: Inference{T}
    ϵ::T #Convergence criteria
    nIter::Integer #Number of steps performed
    optimizer::Optimizer #Learning rate for stochastic updates
    Stochastic::Bool #Use of mini-batches
    nSamples::Int64
    nSamplesUsed::Int64 #Size of mini-batches
    MBIndices::AbstractVector #Indices of the minibatch
    ρ::T #Stochastic Coefficient
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    ∇η₁::AbstractVector{AbstractVector}
    ∇η₂::AbstractVector{AbstractArray}
    function AnalyticInference{T}(ϵ::T,nIter::Integer,optimizer::Optimizer,Stochastic::Bool,nSamples::Integer,ρ::T,flag::Bool) where T
        return new{T}(ϵ,nIter,optimizer,Stochastic,nSamples,ρ,flag)
    end
    function AnalyticInference{T}(ϵ::T,nIter::Integer,optimizer::Optimizer,Stochastic::Bool,nSamples::Integer,ρ::T,flag::Bool,∇η₁::AbstractVector{AbstractVector},
    ∇η₂::AbstractVector{AbstractArray}) where T
        return new{T}(ϵ,nIter,optimizer,Stochastic,nSamples,ρ,flag,∇η₁,∇η₂)
    end
end

function AnalyticInference(nSamples::Integer;ϵ::T=1e-5,optimizer::Optimizer=VanillaGradDescent(η=1.0)) where {T<:Real}
    AnalyticInference{T}(ϵ,0,optimizer,false,nSamples,nSamples,1:nSamples,nSamples/nSamplesUsed,true)
end

function AnalyticInference(nSamples::Integer,nSamplesUsed::Integer,η₁::AbstractVector{AbstractVector},η₂::AbstractVector{AbstractArray};ϵ::T=1e-5,optimizer::Optimizer=Adam(α=0.1)) where {T<:Real}
    AnalyticInference{T}(ϵ,0,optimizer,true,nSamples,nSamplesUsed,1:nSamplesUsed,nSamples/nSamplesUsed,true,similar(η₁),similar(η₂))
end

function AnalyticInference(;ϵ::T=1e-5,optimizer::Optimizer=VanillaGradDescent(η=1.0)) where {T<:Real}
    AnalyticInference{Float64}(ϵ,0,optimizer,false,1,1,1.0,true)
end

function variational_updates!(model::VGP{L,AnalyticInference{T}}) where {L<:Likelihood,T}
    local_updates!(model)
    natural_gradient!(model)
    global_update!(model)
end

function variational_updates(model::SVGP{L,AnalyticInference}) where {L<:Likelihood}
    local_updates!(model)
    natural_gradient!(model)
    computeLearningRate(model)
    global_update!(model)
end

function global_update!(model::VGP)
    model.Σ .= inv.(model.η₂)*(-0.5)
    model.μ .= model.Σ.*model.η₁
end
