include("vi_optimizers.jl")
include("analytic.jl")
include("analyticVI.jl")
include("numericalVI.jl")
include("sampling.jl")
include("optimisers.jl")

export RobbinsMonro, ALRSVI

post_process!(::AbstractGPModel) = nothing

# Utils to iterate over inference objects
Base.length(::AbstractInference) = 1
Base.size(::AbstractInference) = (1,)

Base.iterate(i::AbstractInference) = (i, nothing)
Base.iterate(::AbstractInference, ::Any) = nothing

is_stochastic(i::AbstractInference) = i.stoch

## Multiple accessors
conv_crit(i::AbstractInference) = i.ϵ

## Conversion from natural to standard distribution parameters ##
function global_update!(gp::AbstractLatent)
    gp.post.Σ .= -inv(nat2(gp)) / 2
    return gp.post.μ .= cov(gp) * nat1(gp)
end

## For the online case, the size may vary and inplace updates are note valid
function global_update!(gp::OnlineVarLatent)
    gp.post.Σ = -inv(nat2(gp)) / 2
    return gp.post.μ = cov(gp) * nat1(gp)
end

## Default function for getting a view on y
batchsize(inf::AbstractInference) = inf.batchsize
set_batchsize!(inf::AbstractInference, n::Int) = inf.batchsize = n

ρ(inf::AbstractInference) = inf.ρ
ρ(m::AbstractGPModel) = ρ(inference(m))
set_ρ!(inf::AbstractInference, ρ) = inf.ρ = ρ
set_ρ!(m::AbstractGPModel, ρ) = set_ρ!(inference(m), ρ)

setHPupdated!(inf::AbstractInference, status::Bool) = inf.HyperParametersUpdated = status
isHPupdated(inf::AbstractInference) = inf.HyperParametersUpdated

n_iter(inf::AbstractInference) = inf.n_iter

opt(::AbstractInference) = nothing
opt(i::VariationalInference) = i.vi_opt
opt(i::SamplingInference) = i.opt
