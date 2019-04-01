mutable struct StreamingVI{T<:Real} <: Inference{T}
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
    function StreamingVI{T}(ϵ::T,nIter::Integer,optimizer_η₁::AbstractVector{<:Optimizer},optimizer_η₂::AbstractVector{<:Optimizer},Stochastic::Bool,nSamples::Integer,nSamplesUsed::Integer,MBIndices::AbstractVector,ρ::T,flag::Bool) where T
        return new{T}(ϵ,nIter,optimizer_η₁,optimizer_η₂,Stochastic,nSamples,nSamplesUsed,MBIndices,ρ,flag)
    end
    function StreamingVI{T}(ϵ::T,nIter::Integer,optimizer_η₁::AbstractVector{<:Optimizer},optimizer_η₂::AbstractVector{<:Optimizer},Stochastic::Bool,nSamples::Integer,nSamplesUsed::Integer,MBIndices::AbstractVector,ρ::T,flag::Bool,∇η₁::AbstractVector{<:AbstractVector},
    ∇η₂::AbstractVector{<:AbstractMatrix}) where T
        return new{T}(ϵ,nIter,optimizer_η₁,optimizer_η₂,Stochastic,nSamples,nSamplesUsed,MBIndices,ρ,flag,∇η₁,∇η₂)
    end
end

function Base.show(io::IO,inference::StreamingVI{T}) where T
    print(io,"Streaming Variational Inference")
end


"""Initialize the final version of the inference object"""
function init_inference(inference::StreamingVI{T},nLatent::Integer,nFeatures::Integer,nSamples::Integer,nSamplesUsed::Integer) where {T<:Real}
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

"""Generic method for variational updates using analytical formulas"""
function variational_updates!(model::AbstractGP{L,StreamingVI{T}}) where {L<:Likelihood,T}
    # local_updates!(model)
    # natural_gradient!(model)
    # global_update!(model)
end
