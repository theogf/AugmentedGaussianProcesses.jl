abstract type RegressionLikelihood{T<:Real} <: Likelihood{T} end
abstract type ClassificationLikelihood{T<:Real} <: Likelihood{T} end

include("gaussian.jl")
include("studentt.jl")
include("logistic.jl")
include("bayesiansvm.jl")
include("multiclass.jl")

function pdf(l::Likelihood{T},y::Real,f::Real) where T
    @error "pdf not implemented"
end

function logpdf(l::Likelihood{T},y::Real,f::Real) where T
    log(pdf(l,y,f))
end

isaugmented(::Likelihood{T}) where T = false
