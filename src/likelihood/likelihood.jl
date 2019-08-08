include("regression.jl")
include("classification.jl")
include("multiclass.jl")
include("event.jl")
include("generic_likelihood.jl")
function pdf(l::Likelihood{T},y::Real,f::Real) where T
    @error "pdf not implemented"
end

@inline logpdf(l::Likelihood{T},y::Real,f::Real) where T = log(pdf(l,y,f))

isaugmented(::Likelihood) = false

Base.length(::Likelihood) = 1

Base.iterate(l::Likelihood) = (l,nothing)
Base.iterate(l::Likelihood, ::Any) = nothing
