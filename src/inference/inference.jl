
include("learning_rate/alrsvi.jl")
include("learning_rate/inversedecay.jl")
include("analytic.jl")
include("gibbssampling.jl")
include("numerical.jl")

function post_process!(model::AbstractGP{<:Likelihood,<:Inference})
    nothing
end
