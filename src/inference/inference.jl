
include("learning_rate/learning_rate.jl")

abstract type Inference{T<:Real} end

""" Solve conjugate or conditionally conjugate likelihoods (especially valid for augmented likelihoods) """
struct AnalyticInference{T<:Real} <: Inference{T}
    ϵ::T #Convergence criteria
    nIter::Integer #Number of steps performed
    λ::LearningRate{T} #Learning rate for stochastic updates
    ρ::T #Stochastic Coefficient
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    ∇η₁::AbstractVector{AbstractVector}
    ∇η₂::AbstractVector{AbstractArray}
end

function AnalyticInference(;ϵ::T=1e-5,learning_rate::LearningRate=DummyLearningRate(1.0)) where {T<:Real}
    AnalyticInference{T}(ϵ,0,learning_rate,1.0,true)
end

include("analytic.jl")
include("gibbssampling.jl")


function post_process!(model::GP{<:Likelihood,I<:Inference})
end
