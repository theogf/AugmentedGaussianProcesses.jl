include("poisson.jl")
include("negativebinomial.jl")

const EventLikelihood = Union{PoissonLikelihood,NegBinomialLikelihood}

""" Return the labels in a vector of vectors for multiple outputs"""
function treat_labels!(y::AbstractVector{<:Int}, ::EventLikelihood)
    return y
end

function treat_labels!(::AbstractVector{<:Real}, ::EventLikelihood)
    return error("For event count target(s) should be integers")
end
