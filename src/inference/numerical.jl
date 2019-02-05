""" Solve any non-conjugate likelihood
by making a numerical approximation (quadrature or MC integration)
of the expected log-likelihood"""
mutable struct NumericalInference{T<:Real} <: Inference{T}
    ϵ::T #Convergence criteria
    nIter::Integer #Number of steps performed
    optimizer::Optimizer #Learning rate for stochastic updates
    nMC::Int64 #Number of samples for MC Integrations
    Stochastic::Bool #Use of mini-batches
    nSamples::Int64
    nSamplesUsed::Int64 #Size of mini-batches
    MBIndices::AbstractVector #Indices of the minibatch
    ρ::T #Stochastic Coefficient
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    ∇η₁::AbstractVector{AbstractVector}
    ∇η₂::AbstractVector{AbstractArray}
    function NumericalInference{T}(ϵ::T,nMC::Integer,nIter::Integer,optimizer::Optimizer,Stochastic::Bool,nSamples::Integer,nSamplesUsed::Integer,MBIndices::AbstractVector,ρ::T,flag::Bool) where T
        return new{T}(ϵ,nIter,optimizer,nMC,Stochastic,nSamples,nSamplesUsed,MBIndices,ρ,flag)
    end
    function NumericalInference{T}(ϵ::T,nMC::Integer,nIter::Integer,optimizer::Optimizer,Stochastic::Bool,nSamples::Integer,nSamplesUsed::Integer,MBIndices::AbstractVector,ρ::T,flag::Bool,∇η₁::AbstractVector{<:AbstractVector},
    ∇η₂::AbstractVector{<:AbstractMatrix}) where T
        return new{T}(ϵ,nIter,optimizer,nMC,Stochastic,nSamples,nSamplesUsed,MBIndices,ρ,flag,∇η₁,∇η₂)
    end
end

function NumericalInference(nSample::Integer;ϵ::T=1e-5,nMC::Integer=200,optimizer::Optimizer=VanillaGradDescent(η=1.0)) where {T<:Real}
    NumericalInference{T}(ϵ,nMC,0,optimizer,false,nSample,nSample,1:nSample,1.0,true)
end

function NumericalInference(Stochastic::Bool,nSample::Integer,nSampleUsed::Integer,η₁::AbstractVector{<:AbstractVector},η₂::AbstractVector{<:AbstractMatrix};ϵ::T=1e-5,nMC::Integer=200,optimizer::Optimizer=Adam(α=0.1)) where {T<:Real}
    NumericalInference{T}(ϵ,nMC,0,optimizer,Stochastic,nSample,nSampleUsed,1:nSampleUsed,nSample/nSampleUsed,true,copy(η₁),copy(η₂))
end

function NumericalInference(;ϵ::T=1e-5,nMC::Integer=200,optimizer::Optimizer=VanillaGradDescent(η=1.0)) where {T<:Real}
    NumericalInference{Float64}(ϵ,nMC,0,optimizer,false,1,1,[1],1.0,true)
end

function init_inference(inference::NumericalInference{T},nLatent::Integer,nFeatures::Integer,nSamplesUsed::Integer) where {T<:Real}
    return inference
end

function variational_updates!(model::VGP{L,NumericalInference{T}}) where {L<:Likelihood,T}
    compute_grad_expectations!(model)
    natural_gradient!(model)
    global_update!(model)
end

function variational_updates!(model::SVGP{L,NumericalInference{T}}) where {L<:Likelihood,T}
    compute_grad_expectations!(model)
    natural_gradient!(model)
    compute_learningrate!(model)
    global_update!(model)
end

#TODO Make two possible cases : MCMC or Quadrature
function compute_grad_expectations(model::VGP)
    for k in model.nLatent
        for i in 1:model.nSample
            model.inference.∇η₁[k][i],model.inference.∇η₂[k][i,i] = loglike_expectations(x->grad_logpdf(model.likelihood,model.y[k][i],x),Normal(μ[i],sqrt(Σ[i,i])))
        end
    end
end

function natural_gradient!(model::VGP) where T
    model.inference.∇η₁ .= expec_μ(model) .- model.η₁
    model.inference.∇η₂ .= Symmetric.(-Diagonal.(expec_Σ(model))+0.5.*model.invKmm .- model.η₂)
end

function natural_gradient!(model::SVGP) where T
    model.inference.∇η₁ .= model.inference.ρ.*transpose.(model.κ).*expec_μ(model) .- model.η₁
    model.inference.∇η₂ .= Symmetric.(-model.inference.ρ.*transpose.(model.κ).*Diagonal.(expec_Σ(model)).*model.κ.+0.5.*model.invKmm .- model.η₂)
end

function global_update!(model::VGP)
    model.Σ .= inv.(model.η₂)*(-0.5)
    model.μ .= model.Σ.*model.η₁
end

function

function compute_learningrate!(model::SVGP{L,NumericalInference{T}}) where {L<:Likelihood,T}
 #TODO learningrate_optimizer
end

function global_update!(model::SVGP{L,NumericalInference{T}}) where {L<:Likelihood,T}
    if model.inference.Stochastic
    else
        model.η₁ .= model.inference.∇η₁ .+ model.η₁
        model.η₂ .= Symmetric.(model.inference.∇η₂ .+ model.η₂)
    end
    model.Σ .= inv.(model.η₂)*(-0.5)
    model.μ .= model.Σ.*model.η₁
end
