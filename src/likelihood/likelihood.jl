include("gaussian.jl")
include("studentt.jl")
include("logistic.jl")
include("bayesiansvm.jl")
include("multiclass.jl")

function pdf(l::Likelihood,y::Real,f::Real)
    @error "pdf not implemented"
end

function logpdf(l::Likelihood,y::Real,f::Real)
    log(pdf(l,y,f))
end
