include("vi_optimizers.jl")
include("analytic.jl")
include("analyticVI.jl")
include("numericalVI.jl")
include("sampling.jl")
include("optimisers.jl")

export RobbinsMonro, ALRSVI

post_process!(model::AbstractGP) = nothing

# Utils to iterate over inference objects
Base.length(::Inference) = 1

Base.iterate(i::Inference) = (i, nothing)
Base.iterate(i::Inference, ::Any) = nothing

isStochastic(i::Inference) = i.stoch

## Multiple accessors
conv_crit(i::Inference) = i.ϵ


## Conversion from natural to standard distribution parameters ##
function global_update!(gp::AbstractLatent)
    gp.post.Σ .= -0.5 * inv(nat2(gp))
    gp.post.μ .= cov(gp) * nat1(gp)
end

## For the online case, the size may vary and inplace updates are note valid
function global_update!(gp::OnlineVarLatent) where {T,L}
    gp.post.Σ = -0.5 * inv(nat2(gp))
    gp.post.μ = cov(gp) * nat1(gp)
end


## Default function for getting a view on y
xview(inf::Inference) = inf.xview
setxview!(inf::Inference, xview) = inf.xview = xview

setyview!(inf::Inference, yview) = inf.yview = yview
yview(inf::Inference) = inf.yview

nMinibatch(inf::Inference) = inf.nMinibatch
setnMinibatch!(inf::Inference, n::Int) = inf.nMinibatch = n

nSamples(i::Inference) = i.nSamples
setnSamples!(inf::Inference, n::Int) = inf.nSamples = n

getρ(inf::Inference) = inf.ρ
setρ!(inf::Inference, ρ) = inf.ρ = ρ

MBIndices(inf::Inference) = inf.MBIndices
setMBIndices!(inf::Inference, mbindices::AbstractVector) =
    inf.MBIndices .= mbindices

setHPupdated!(inf::Inference, status::Bool) =
    inf.HyperParametersUpdated = status
isHPupdated(inf::Inference) = inf.HyperParametersUpdated

nIter(inf::Inference) = inf.nIter

get_opt(i::Inference) = nothing
get_opt(i::VariationalInference) = i.vi_opt
get_opt(i::SamplingInference) = i.opt
get_opt(i::Inference, n::Int) = get_opt(i)[n]
opt_type(i::Inference) = first(get_opt(i))

function tuple_inference(
    i::Inference,
    nLatent::Int,
    nFeatures::Int,
    nSamples::Int,
    nMinibatch::Int,
    xview::AbstractVector,
    yview::AbstractVector
)
    return tuple_inference(
        i,
        nLatent,
        fill(nFeatures, nLatent),
        nSamples,
        nMinibatch,
        xview,
        yview
    )
end
