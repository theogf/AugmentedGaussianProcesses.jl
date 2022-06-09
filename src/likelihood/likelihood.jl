include("regression.jl")
include("classification.jl")
include("multiclass.jl")
include("event.jl")
# include("generic_likelihood.jl")
function (l::AbstractLikelihood)(::Real, ::Real)
    return error("pdf not implemented for likelihood $(typeof(l))")
end

Distributions.loglikelihood(l::AbstractLikelihood, y::Real, f) = log(l(y, f))

## Default function for getting gradient ##
function ∇loglikehood(l::AbstractLikelihood, y::Real, f::Real)
    return only(ForwardDiff.gradient(x -> loglikelihood(l, y, x[1]), [f]))
end

function ∇loglikehood(l::AbstractLikelihood, y::Real, f::AbstractVector)
    return ForwardDiff.gradient(x -> loglikelihood(l, y, x), f)
end

function hessloglikehood(l::AbstractLikelihood, y::Real, f::Real)
    return only(ForwardDiff.hessian(x -> loglikelihood(l, y, x[1]), [f]))
end

function hessloglikelihood(l::AbstractLikelihood, y::Real, f::AbstractVector)
    return ForwardDiff.hessian(x -> loglikelihood(l, y, x), f)
end

implemented(::AbstractLikelihood, ::AbstractInference) = false

n_latent(::AbstractLikelihood) = 1

# Allows to use likelihoods in broadcasts
Base.length(::AbstractLikelihood) = 1
Base.iterate(l::AbstractLikelihood) = (l, nothing)
Base.iterate(::AbstractLikelihood, ::Any) = nothing
