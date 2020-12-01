include("regression.jl")
include("classification.jl")
include("multiclass.jl")
include("event.jl")
# include("generic_likelihood.jl")
function (l::Likelihood)(y::Real,f::Real)
    error("pdf not implemented for likelihood $(typeof(l))")
end

Distributions.loglikelihood(l::Likelihood, y::Real, f) = log(l(y,f))

## Default function for getting gradient ##
function grad_loglike(l::Likelihood, y::Real, f::Real)
    first(ForwardDiff.gradient(x->loglikelihood(l, y, x[1]), [f]))
end

function grad_loglike(l::Likelihood, y::Real, f::AbstractVector)
    ForwardDiff.gradient(x->loglikelihood(l, y, x), f)
end

function hessian_loglike(l::Likelihood, y::Real, f::Real)
    first(ForwardDiff.hessian(x->loglikelihood(l, y, x[1]), [f]))
end

function hessian_loglike(l::Likelihood,y::Real,f::AbstractVector)
    ForwardDiff.hessian(x->loglikelihood(l, y, x), f)
end

implemented(::Likelihood,::Inference) = false

isaugmentable(::Likelihood) = false

Base.length(::Likelihood) = 1

num_latent(::Likelihood) = 1

Base.iterate(l::Likelihood) = (l, nothing)
Base.iterate(l::Likelihood, ::Any) = nothing
