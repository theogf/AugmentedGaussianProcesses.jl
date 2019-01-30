
include("learning_rate/learning_rate.jl")
include("analytic.jl")
include("gibbssampling.jl")


function post_process!(model::GP{<:Likelihood,<:Inference})
end
