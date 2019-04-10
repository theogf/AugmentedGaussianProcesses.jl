abstract type EventLikelihood{T<:Real} <: Likelihood{T} end

include("poisson.jl")
