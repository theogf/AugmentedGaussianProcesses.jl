include("vi_optimizers.jl")
include("analytic.jl")
include("analyticVI.jl")
include("numericalVI.jl")
include("sampling.jl")
include("optimisers.jl")

export RobbinsMonro, ALRSVI

post_process!(::AbstractGP) = nothing

# Utils to iterate over inference objects
Base.length(::AbstractInference) = 1

Base.iterate(i::AbstractInference) = (i, nothing)
Base.iterate(::AbstractInference, ::Any) = nothing

isStochastic(i::AbstractInference) = i.stoch

## Multiple accessors
conv_crit(i::AbstractInference) = i.ϵ


## Conversion from natural to standard distribution parameters ##
function global_update!(gp::AbstractLatent)
    gp.post.Σ .= -0.5 * inv(nat2(gp))
    gp.post.μ .= cov(gp) * nat1(gp)
end

## For the online case, the size may vary and inplace updates are note valid
function global_update!(gp::OnlineVarLatent)
    gp.post.Σ = -0.5 * inv(nat2(gp))
    gp.post.μ = cov(gp) * nat1(gp)
end


## Default function for getting a view on y
xview(inf::AbstractInference) = inf.xview
setxview!(inf::AbstractInference, xview) = inf.xview = xview

setyview!(inf::AbstractInference, yview) = inf.yview = yview
yview(inf::AbstractInference) = inf.yview

nMinibatch(inf::AbstractInference) = inf.nMinibatch
setnMinibatch!(inf::AbstractInference, n::Int) = inf.nMinibatch = n

nSamples(i::AbstractInference) = i.nSamples
setnSamples!(inf::AbstractInference, n::Int) = inf.nSamples = n

getρ(inf::AbstractInference) = inf.ρ
setρ!(inf::AbstractInference, ρ) = inf.ρ = ρ

MBIndices(inf::AbstractInference) = inf.MBIndices
setMBIndices!(inf::AbstractInference, mbindices::AbstractVector) =
    inf.MBIndices .= mbindices

setHPupdated!(inf::AbstractInference, status::Bool) =
    inf.HyperParametersUpdated = status
isHPupdated(inf::AbstractInference) = inf.HyperParametersUpdated

nIter(inf::AbstractInference) = inf.nIter

get_opt(::AbstractInference) = nothing
get_opt(i::VariationalInference) = i.vi_opt
get_opt(i::SamplingInference) = i.opt
get_opt(i::AbstractInference, n::Int) = get_opt(i)[n]
opt_type(i::AbstractInference) = first(get_opt(i))

# Initialize the final version of the inference objects
# using the right parametrization and size
function tuple_inference(
    i::AbstractInference,
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
