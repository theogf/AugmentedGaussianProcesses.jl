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
    function AnalyticInference{T}(ϵ::T,nIter::Integer,optimizer::Optimizer,Stochastic::Bool,nSamples::Integer,nSamplesUsed::Integer,MBIndices::AbstractVector,ρ::T,flag::Bool) where T
        return new{T}(ϵ,nIter,optimizer,Stochastic,nSamples,nSamplesUsed,MBIndices,ρ,flag)
    end
    function AnalyticInference{T}(ϵ::T,nIter::Integer,optimizer::Optimizer,Stochastic::Bool,nSamples::Integer,nSamplesUsed::Integer,MBIndices::AbstractVector,ρ::T,flag::Bool,∇η₁::AbstractVector{<:AbstractVector},
    ∇η₂::AbstractVector{<:AbstractMatrix}) where T
        return new{T}(ϵ,nIter,optimizer,Stochastic,nSamples,nSamplesUsed,MBIndices,ρ,flag,∇η₁,∇η₂)
    end
end

function AnalyticInference(nSample::Integer;ϵ::T=1e-5,optimizer::Optimizer=VanillaGradDescent(η=1.0)) where {T<:Real}
    AnalyticInference{T}(ϵ,0,optimizer,false,nSample,nSample,1:nSample,1.0,true)
end

function AnalyticInference(Stochastic::Bool,nSample::Integer,nSampleUsed::Integer,η₁::AbstractVector{<:AbstractVector},η₂::AbstractVector{<:AbstractMatrix};ϵ::T=1e-5,optimizer::Optimizer=Adam(α=0.1)) where {T<:Real}
    AnalyticInference{T}(ϵ,0,optimizer,Stochastic,nSample,nSampleUsed,1:nSampleUsed,nSample/nSampleUsed,true,similar(η₁),similar(η₂))
end

function AnalyticInference(;ϵ::T=1e-5,optimizer::Optimizer=VanillaGradDescent(η=1.0)) where {T<:Real}
    AnalyticInference{Float64}(ϵ,0,optimizer,false,1,1,[1],1.0,true)
end

function init_inference(inference::AnalyticInference{T},nLatent::Integer,nFeatures::Integer,nSamplesUsed::Integer) where {T<:Real}
    inference.∇η₁ = [zeros(T,nFeatures) for _ in 1:nLatent];
    inference.∇η₂ = [Symmetric(Diagonal(ones(T,nFeatures))) for _ in 1:nLatent]
    return inference
end

function variational_updates!(model::VGP{L,AnalyticInference{T}}) where {L<:Likelihood,T}
    local_updates!(model)
    natural_gradient!(model)
    global_update!(model)
end

function variational_updates!(model::SVGP{L,AnalyticInference{T}}) where {L<:Likelihood,T}
    local_updates!(model)
    natural_gradient!(model)
    compute_learningrate!(model)
    global_update!(model)
end

function natural_gradient!(model::VGP{L,AnalyticInference{T}}) where {L<:Likelihood,T}
    model.η₁ .+= model.inference.∇η₁ .= expec_μ(model) .- model.η₁
    model.η₂ = Symmetric.((model.inference.∇η₂ .= Symmetric.(-Diagonal.(expec_Σ(model))-0.5.*model.invKnn .- model.η₂)) .+ model.η₂)
end

function natural_gradient!(model::SVGP{L,AnalyticInference{T}}) where {L<:Likelihood,T}
    model.inference.∇η₁ .= model.inference.ρ.*transpose.(model.κ).*expec_μ(model) .- model.η₁
    model.inference.∇η₂ .= Symmetric.(-model.inference.ρ.*transpose.(model.κ).*Diagonal.(expec_Σ(model)).*model.κ.-0.5.*model.invKmm .- model.η₂)
end

function global_update!(model::VGP{L,AnalyticInference{T}}) where {L<:Likelihood,T}
    model.Σ .= inv.(model.η₂)*(-0.5)
    model.μ .= model.Σ.*model.η₁
end


function compute_learningrate!(model::SVGP{L,AnalyticInference{T}}) where {L<:Likelihood,T}
 #TODO learningrate_optimizer
end

function global_update!(model::SVGP{L,AnalyticInference{T}}) where {L<:Likelihood,T}
    if model.inference.Stochastic
    else
        model.η₁ .= model.inference.∇η₁ .+ model.η₁
        model.η₂ .= Symmetric.(model.inference.∇η₂ .+ model.η₂)
    end
    model.Σ .= inv.(model.η₂)*(-0.5)
    model.μ .= model.Σ.*model.η₁
end
