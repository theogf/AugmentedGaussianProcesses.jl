
include("learning_rate/alrsvi.jl")
include("learning_rate/inversedecay.jl")
include("vi_optimizers.jl")
include("analytic.jl")
include("analyticVI.jl")
include("gibbssampling.jl")
include("numericalVI.jl")

function post_process!(model::AbstractGP{T,<:Likelihood,<:Inference}) where {T}
    nothing
end

Base.length(::Inference) = 1

Base.iterate(l::Inference) = (l,nothing)
Base.iterate(l::Inference, ::Any) = nothing


const GibbsorVI = Union{<:GibbsSampling,<:AnalyticVI}
