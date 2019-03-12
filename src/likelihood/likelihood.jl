include("regression.jl")
include("classification.jl")
include("multiclass.jl")

function pdf(l::Likelihood{T},y::Real,f::Real) where T
    @error "pdf not implemented"
end

function logpdf(l::Likelihood{T},y::Real,f::Real) where T
    log(pdf(l,y,f))
end

isaugmented(::Likelihood{T}) where T = false
