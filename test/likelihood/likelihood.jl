struct NewLikelihood{T} <: AGP.Likelihood{T} end
@test_throws ErrorException AGP.pdf(NewLikelihood{Float64}(), rand(), rand())
@test length(NewLikelihood{Float64}()) == 1
