abstract type MultiClassLikelihood{T<:Real} <: Likelihood{T} end

function init_multiclass_likelihood(likelihood::L,y::AbstractVector) where {L<:MultiClassLikelihood{T} where T}
    L(one_of_K_mapping(y))
end

abstract type AbstractLogisticSoftMaxLikelihood{T<:Real} <: MultiClassLikelihood{T} end


include("softmax.jl")
include("logisticsoftmax.jl")
