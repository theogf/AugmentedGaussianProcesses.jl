include("regression.jl")
include("classification.jl")
include("multiclass.jl")
include("event.jl")
# include("generic_likelihood.jl")
include("customlikelihood.jl")
function pdf(l::Likelihood{T},y::Real,f::Real) where {T}
    throw(ErrorException("pdf not implemented for likelihood $(typeof(l))"))
end

@inline logpdf(l::Likelihood{T},y::Real,f::Real) where {T} = log(pdf(l,y,f))
logpdf(l::Likelihood{T},y::Real,f::AbstractVector) where {T} = log(pdf(l,y,f))

implemented(::Likelihood,::Inference) = false

isaugmentable(::Likelihood) = false

Base.length(::Likelihood) = 1

num_latent(::Likelihood) = 1

Base.iterate(l::Likelihood) = (l,nothing)
Base.iterate(l::Likelihood, ::Any) = nothing
