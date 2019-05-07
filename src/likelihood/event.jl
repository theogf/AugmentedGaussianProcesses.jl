abstract type EventLikelihood{T<:Real} <: Likelihood{T} end

include("poisson.jl")


""" Return the labels in a vector of vectors for multiple outputs"""
function treat_labels!(y::AbstractArray{T,N},likelihood::L) where {T,N,L<:EventLikelihood}
    @assert T<:Integer "For event count target(s) should be integers (Bool,Integer or Float)"
    @assert N <= 2 "Target should be a matrix or a vector"
    if N == 1
        return [y],1,likelihood
    else
        return [y[:,i] for i in 1:size(y,2)],size(y,2),likelihood
    end
end
