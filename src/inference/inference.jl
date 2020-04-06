include("vi_optimizers.jl")
include("analytic.jl")
include("analyticVI.jl")
include("numericalVI.jl")
include("sampling.jl")
include("optimisers.jl")

export RobbinsMonro, ALRSVI

function post_process!(model::AbstractGP{T,<:Likelihood,<:Inference}) where {T}
    nothing
end

Base.length(::Inference) = 1

Base.iterate(l::Inference) = (l,nothing)
Base.iterate(l::Inference, ::Any) = nothing

isStochastic(l::Inference) = l.Stochastic

## Conversion from natural to standard distribution parameters ##
function global_update!(gp::Abstract_GP) where {T,L}
    gp.Σ .= -0.5*inv(gp.η₂)
    gp.μ .= gp.Σ*gp.η₁
end

## For the online case, the size may vary and inplace updates are note valid
function global_update!(gp::_OSVGP) where {T,L}
    gp.Σ = -0.5*inv(gp.η₂)
    gp.μ = gp.Σ*gp.η₁
end


## Default function for getting a view on y
xview(inf::Inference, i::Int) = inf.xview[i]
xview(inf::Inference) = xview(inf, 1)
yview(inf::Inference, i::Int) = inf.yview[i]
yview(inf::Inference) = yview(inf, 1)

setxview!(inf::Inference, i::Int, xview) = inf.xview[i] = xview
setxview!(inf::Inference, xview) = setxview!(inf, 1, xview)
setyview!(inf::Inference, i::Int, yview) = inf.yview[i] = yview
setyview!(inf::Inference, yview) = setyview!(inf, 1, yview)

nMinibatch(inf::Inference, i::Int) = inf.nMinibatch[i]
nMinibatch(inf::Inference) = nMinibatch(inf, 1)
setnMinibatch!(inf::Inference, n::AbstractVector) = inf.nMinibatch = n
setnMinibatch!(inf::Inference, n::Real) = setnMinibatch!(inf,[n])

setnSamples!(inf::Inference, n::AbstractVector) = inf.nSamples = n
setnSamples!(inf::Inference, n::Real) = setnSamples!(inf,[n])

getρ(inf::Inference, i::Int) = inf.ρ[i]
getρ(inf::Inference) = getρ(inf, 1)

MBIndices(inf::Inference, i::Int) = inf.MBIndices[i]
MBIndices(inf::Inference) = MBIndices(inf, 1)
setMBIndices!(inf::Inference, i::Int, mbindices::AbstractVector) = inf.MBIndices[i] .= mbindices
setMBIndices!(inf::Inference, mbindices::AbstractVector) = setMBIndices!(inf, 1, mbindices)

setHPupdated!(inf::Inference, status::Bool) = inf.HyperParametersUpdated = status
isHPupdated(inf::Inference) = inf.HyperParametersUpdated

nIter(inf::Inference) = inf.nIter

@inline view_y(l::Likelihood, y::AbstractVector, i::AbstractVector) = view(y, i)

get_opt(i::Inference) = nothing
get_opt(i::VariationalInference) = i.vi_opt
get_opt(i::SamplingInference) = i.opt
get_opt(i::Inference, n::Int) = get_opt(i)[n]
opt_type(i::Inference) = first(get_opt(i))

# isStochastic(inf::Inference) = inf.Stochastic

function tuple_inference(
    i::Inference,
    nLatent::Integer,
    nFeatures::Integer,
    nSamples::Integer,
    nMinibatch::Integer,
)
    return tuple_inference(
        i,
        nLatent,
        fill(nFeatures, nLatent),
        fill(nSamples, nLatent),
        fill(nMinibatch, nLatent),
    )
end
