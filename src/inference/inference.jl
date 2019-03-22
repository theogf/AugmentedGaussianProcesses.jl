
include("learning_rate/alrsvi.jl")
include("learning_rate/inversedecay.jl")
include("analytic.jl")
include("analyticVI.jl")
include("gibbssampling.jl")
include("numericalVI.jl")

function post_process!(model::AbstractGP{<:Likelihood,<:Inference})
    nothing
end
