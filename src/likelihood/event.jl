abstract type EventLikelihood{T<:Real} <: Likelihood{T} end

include("poisson.jl")
include("negativebinomial.jl")

""" Return the labels in a vector of vectors for multiple outputs"""
function treat_labels!(y::AbstractVector{T},likelihood::L) where {T,N,L<:EventLikelihood}
    @assert T<:Integer "For event count target(s) should be integers"
    return y,1,likelihood
end
