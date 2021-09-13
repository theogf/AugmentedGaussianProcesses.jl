abstract type ClassificationLikelihood{T<:Real} <: AbstractLikelihood{T} end

include("logistic.jl")
include("bayesiansvm.jl")

""" Return the labels in a vector of vectors for multiple outputs"""
function treat_labels!(y::AbstractVector{<:Real}, ::ClassificationLikelihood)
    labels = unique(y)
    if sort(Int64.(labels)) == [0; 1]
        return (y .- 0.5) * 2
    elseif sort(Int64.(labels)) == [-1; 1]
        return y
    else
        throw(ArgumentError("Labels of y should be binary {-1,1} or {0,1}"))
    end
end

function treat_labels!(::AbstractVector, ::ClassificationLikelihood)
    return error(
        "For classification target(s) should be real valued (Bool, Integer or Float)"
    )
end

predict_y(::ClassificationLikelihood, μ::AbstractVector{<:Real}) = μ .> 0
predict_y(::ClassificationLikelihood, μ::AbstractVector{<:AbstractVector}) = first(μ) .> 0
