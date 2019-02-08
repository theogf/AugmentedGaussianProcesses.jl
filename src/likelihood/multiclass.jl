abstract type MultiClassLikelihood{T<:Real} <: Likelihood{T} end

""" Return the labels in a vector of vectors for multiple outputs"""
function treat_labels!(y::AbstractArray{T,N},likelihood::L) where {T,N,L<:SoftMaxLikelihood}
    @assert N <= 1 "Target should be a vector of labels"
    likelihood = init_multiclass_likelihood(y,likelihood)
end

function init_multiclass_likelihood(likelihood::L,y::AbstractVector) where {L<:MultiClassLikelihood{T} where T}
    L(one_of_K_mapping(y)...)
end

include("softmax.jl")
include("logisticsoftmax.jl")
