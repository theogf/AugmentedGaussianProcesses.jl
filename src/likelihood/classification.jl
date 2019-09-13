abstract type ClassificationLikelihood{T<:Real} <: Likelihood{T} end

include("logistic.jl")
include("bayesiansvm.jl")

""" Return the labels in a vector of vectors for multiple outputs"""
function treat_labels!(y::AbstractArray{T,N},likelihood::L) where {T,N,L<:ClassificationLikelihood}
    @assert T<:Real "For classification target(s) should be real valued (Bool,Integer or Float)"
    @assert N <= 2 "Target should be a matrix or a vector"
    labels = Int64.(unique(y))
    @assert (length(labels) <= 2) && ((sort(labels) == [0;1]) || (sort(labels) == [-1;1])) "Labels of y should be binary {-1,1} or {0,1}"
    if N == 1
        return [y],1,likelihood
    else
        return [y[:,i] for i in 1:size(y,2)],size(y,2),likelihood
    end
end
