abstract type RegressionLikelihood{T<:Real} <: Likelihood{T} end

include("gaussian.jl")
include("studentt.jl")
include("laplace.jl")
include("heteroscedastic.jl")
include("matern.jl")

""" Return the labels in a vector of vectors for multiple outputs"""
function treat_labels!(y::AbstractArray{T,N},likelihood::L) where {T,N,L<:RegressionLikelihood}
    @assert T<:Real "For regression target(s) should be real valued"
    @assert N <= 2 "Target should be a matrix or a vector"
    if N == 1
        return [y],1,likelihood
    else
        return [y[:,i] for i in 1:size(y,2)],size(y,2),likelihood
    end
end
