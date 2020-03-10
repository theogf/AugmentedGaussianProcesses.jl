abstract type ClassificationLikelihood{T<:Real} <: Likelihood{T} end

include("logistic.jl")
include("bayesiansvm.jl")

""" Return the labels in a vector of vectors for multiple outputs"""
function treat_labels!(y::AbstractVector{T},likelihood::L) where {T,N,L<:ClassificationLikelihood}
    @assert T<:Real "For classification target(s) should be real valued (Bool,Integer or Float)"
    labels = unique(y)
    @assert length(labels) <= 2 "Labels of y should be binary {-1,1} or {0,1}"
    if sort(Int64.(labels)) == [0;1]
        return (y.-0.5)*2,1,likelihood
    elseif sort(Int64.(labels)) == [-1;1]
        return y,1,likelihood
    else
        throw(AssertionError("Labels of y should be binary {-1,1} or {0,1}"))
    end
end

predict_y(l::ClassificationLikelihood,μ::AbstractVector{<:Real}) = sign.(μ)
predict_y(l::ClassificationLikelihood,μ::AbstractVector{<:AbstractVector}) = sign.(first(μ))
