abstract type EventLikelihood{T<:Real} <: Likelihood{T} end

include("poisson.jl")
include("negativebinomial.jl")

""" Return the labels in a vector of vectors for multiple outputs"""
function treat_labels!(y::AbstractVector{<:Int}, likelihood::EventLikelihood)
    return y, 1, likelihood
end

function treat_labels!(y::AbstractVector{<:Real}, likelihood::EventLikelihood)
    error("For event count target(s) should be integers")
end
