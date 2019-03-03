""" Solve conjugate or conditionally conjugate likelihoods (especially valid for augmented likelihoods) """
mutable struct AnalyticInference{T<:Real} <: Inference{T}
    ϵ::T #Convergence criteria
    nIter::Integer #Number of steps performed
    optimizer_η₁::LatentArray{Optimizer} #Learning rate for stochastic updates
    optimizer_η₂::LatentArray{Optimizer} #Learning rate for stochastic updates
    Stochastic::Bool #Use of mini-batches
    nSamples::Int64
    nSamplesUsed::Int64 #Size of mini-batches
    MBIndices::Vector{Int64} #Indices of the minibatch
    ρ::T #Stochastic Coefficient
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    ∇η₁::LatentArray{Vector{T}}
    ∇η₂::LatentArray{Matrix{T}} #Stored as a matrix since symmetric sums do not help for the moment WARNING
    function AnalyticInference{T}(ϵ::T,nIter::Integer,optimizer_η₁::AbstractVector{<:Optimizer},optimizer_η₂::AbstractVector{<:Optimizer},Stochastic::Bool,nSamples::Integer,nSamplesUsed::Integer,MBIndices::AbstractVector,ρ::T,flag::Bool) where T
        return new{T}(ϵ,nIter,optimizer_η₁,optimizer_η₂,Stochastic,nSamples,nSamplesUsed,MBIndices,ρ,flag)
    end
    function AnalyticInference{T}(ϵ::T,nIter::Integer,optimizer_η₁::AbstractVector{<:Optimizer},optimizer_η₂::AbstractVector{<:Optimizer},Stochastic::Bool,nSamples::Integer,nSamplesUsed::Integer,MBIndices::AbstractVector,ρ::T,flag::Bool,∇η₁::AbstractVector{<:AbstractVector},
    ∇η₂::AbstractVector{<:AbstractMatrix}) where T
        return new{T}(ϵ,nIter,optimizer_η₁,optimizer_η₂,Stochastic,nSamples,nSamplesUsed,MBIndices,ρ,flag,∇η₁,∇η₂)
    end
end

function StochasticAnalyticInference(nMinibatch::Integer;ϵ::T=1e-5,optimizer::Optimizer=ALRSVI()) where {T<:Real}
# function StochasticAnalyticInference(nMinibatch::Integer;ϵ::T=1e-5,optimizer::Optimizer=Adam(α=0.001)) where {T<:Real}
    AnalyticInference{T}(ϵ,0,[optimizer],[optimizer],true,1,nMinibatch,1:nMinibatch,1.0,true)
end

function AnalyticInference(;ϵ::T=1e-5,optimizer::Optimizer=VanillaGradDescent(η=1.0)) where {T<:Real}
    AnalyticInference{Float64}(ϵ,0,[optimizer],[optimizer],false,1,1,[1],1.0,true)
end

function Base.show(io::IO,inference::AnalyticInference{T}) where T
    print(io,"($(inference.Stochastic ? "Stochastic analytic" : "Analytic") inference")
end

function init_inference(inference::AnalyticInference{T},nLatent::Integer,nFeatures::Integer,nSamples::Integer,nSamplesUsed::Integer) where {T<:Real}
    inference.nSamples = nSamples
    inference.nSamplesUsed = nSamplesUsed
    inference.MBIndices = 1:nSamplesUsed
    inference.ρ = nSamples/nSamplesUsed
    inference.optimizer_η₁ = [copy(inference.optimizer_η₁[1]) for _ in 1:nLatent]
    inference.optimizer_η₂ = [copy(inference.optimizer_η₂[1]) for _ in 1:nLatent]
    inference.∇η₁ = [zeros(T,nFeatures) for _ in 1:nLatent];
    inference.∇η₂ = [Matrix(Diagonal(ones(T,nFeatures))) for _ in 1:nLatent]
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
    global_update!(model)
end

function natural_gradient!(model::VGP{L,AnalyticInference{T}}) where {L<:Likelihood,T}
    model.η₁ .= ∇μ(model)
    model.η₂ .= -Symmetric.(Diagonal.(∇Σ(model)).+0.5.*model.invKnn)
end

function natural_gradient!(model::SVGP{L,AnalyticInference{T}}) where {L<:Likelihood,T}
    model.inference.∇η₁ .= model.inference.ρ.*transpose.(model.κ).*∇μ(model) .- model.η₁
    model.inference.∇η₂ .= -(model.inference.ρ.*transpose.(model.κ).*Diagonal.(∇Σ(model)).*model.κ.+0.5.*model.invKmm) .- model.η₂
end

function global_update!(model::VGP{L,AnalyticInference{T}}) where {L<:Likelihood,T}
    model.Σ .= -0.5.*inv.(model.η₂)
    model.μ .= model.Σ.*model.η₁
end

function global_update!(model::SVGP{L,AnalyticInference{T}}) where {L<:Likelihood,T}
    if model.inference.Stochastic
        model.η₁ .= model.η₁ .+ GradDescent.update.(model.inference.optimizer_η₁,model.inference.∇η₁)
        model.η₂ .= Symmetric.(model.η₂ .+ GradDescent.update.(model.inference.optimizer_η₂,model.inference.∇η₂))
    else
        model.η₁ .+= model.inference.∇η₁
        model.η₂ .= Symmetric.(model.inference.∇η₂ .+ model.η₂)
    end
    model.Σ .= -0.5.*inv.(model.η₂)
    model.μ .= model.Σ.*model.η₁
end
