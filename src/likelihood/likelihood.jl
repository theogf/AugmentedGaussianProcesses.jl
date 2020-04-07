include("regression.jl")
include("classification.jl")
include("multiclass.jl")
include("event.jl")
# include("generic_likelihood.jl")
function pdf(l::Likelihood{T},y::Real,f::Real) where {T}
    throw(ErrorException("pdf not implemented for likelihood $(typeof(l))"))
end

@inline logpdf(l::Likelihood{T},y::Real,f::Real) where {T} = log(pdf(l,y,f))
logpdf(l::Likelihood{T},y::Real,f::AbstractVector) where {T} = log(pdf(l,y,f))

## Default function for getting gradient ##
function grad_logpdf(l::Likelihood,y::Real,f::Real)
    ForwardDiff.gradient(x->AugmentedGaussianProcesses.logpdf(l,y,x[1]),[f])[1]
end

function grad_logpdf(l::Likelihood,y::Real,f::AbstractVector)
    ForwardDiff.gradient(x->AugmentedGaussianProcesses.logpdf(l,y,x),f)
end

function hessian_logpdf(l::Likelihood,y::Real,f::Real)
    ForwardDiff.hessian(x->AugmentedGaussianProcesses.logpdf(l,y,x[1]),[f])[1]
end

function hessian_logpdf(l::Likelihood,y::Real,f::AbstractVector)
    ForwardDiff.hessian(x->AugmentedGaussianProcesses.logpdf(l,y,x),f)
end

implemented(::Likelihood,::Inference) = false

isaugmentable(::Likelihood) = false

Base.length(::Likelihood) = 1

num_latent(::Likelihood) = 1

Base.iterate(l::Likelihood) = (l,nothing)
Base.iterate(l::Likelihood, ::Any) = nothing
