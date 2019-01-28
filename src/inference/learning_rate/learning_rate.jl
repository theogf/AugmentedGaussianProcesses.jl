abstract type LearningRate{T<:Real} end

struct DummyLearningRate{T<:Real} <: LearningRate{T}
    ρ::T
end

function DummyLearningRate(ρ::T) where {T<:Real}
    DummyLearningRate{T}(ρ)
end
