abstract type LearningRate{T<:Real} end

struct DummyLearningRate{T<:Real} <: LearningRate{T}
    ρ::T
end
