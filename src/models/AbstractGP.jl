abstract type AbstractGP{T<:Real,L<:Likelihood{T},I<:Inference{T},N} end

@traitdef IsFull{X}
@traitdef IsMultiOutput{X}
@traitdef IsSparse{X}

Base.eltype(m::AbstractGP{T}) where {T} = T
data(m::AbstractGP) = m.data
likelihood(m::AbstractGP) = m.likelihood
inference(m::AbstractGP) = m.inference
nLatent(m::AbstractGP{<:Real,<:Likelihood,<:Inference, N}) where {N} = N
nOutput(m::AbstractGP) = 1
@traitfn nFeatures(m::TGP) where {TGP <: AbstractGP; !IsSparse{TGP}} =
    nSamples(m)
@traitfn nFeatures(m::TGP) where {TGP <: AbstractGP; IsSparse{TGP}} =
    m.nFeatures
nSamples(m::AbstractGP) = nSamples(data(m))

getf(m::AbstractGP) = m.f
getf(m::AbstractGP, i::Int) = m.f[i]
Base.getindex(m::AbstractGP, i::Int) = getf(m, i)

nIter(m::AbstractGP) = nIter(inference(m))

input(m::AbstractGP) = input(data(m))
output(m::AbstractGP) = output(data(m))

xview(m::AbstractGP) = xview(inference(m))
yview(m::AbstractGP) = yview(inference(m))

MBIndices(m::AbstractGP) = MBIndices(inference(m))

is_trained(m::AbstractGP) = m.trained
set_trained!(m::AbstractGP, status::Bool) = m.trained = status

verbose(m::AbstractGP) = m.verbose

post_step!(m::AbstractGP) = nothing

function Random.rand!(
    model::AbstractGP,
    A::DenseArray{T},
    X::AbstractVector,
) where {T<:Real}
    rand!(MvNormal(predict_f(model, X, covf = true, diag = false)...), A)
end

Random.rand(model::AbstractGP, X::AbstractMatrix, n::Int = 1) = rand(model, KernelFunctions.RowVecs(X), n)

function Random.rand(
    model::AbstractGP{T},
    X::AbstractVector,
    n::Int = 1,
) where {T<:Real}
    if nLatent(model) == 1
        rand!(model, Array{T}(undef, length(X), n), X)
    else
        @error "Sampling not implemented for multiple output GPs"
    end
end

# Statistics.mean(model::AbstractGP) = model.μ

pr_covs(model::AbstractGP) = pr_cov.(model.f)

means(model::AbstractGP) = mean.(model.f)

covs(model::AbstractGP) = cov.(model.f)

kernels(model::AbstractGP) = kernel.(model.f)


## TODO this should probably be moved to InducingPoints.jl

function setZ!(
    m::AbstractGP,
    Z::AbstractVector{<:AbstractVector{<:Real}},
    i::Int,
)
    @assert size(Z) == size(m.f[i].Z) "Size of Z $(size(Z)) is not the same as in the model $(size(m.f[i].Z))"
    m.f[i].Z.Z = copy(Z)
end

function setZ!(
    m::AbstractGP,
    Z::AbstractVector{<:AbstractVector},
)
    @assert length(Z) == nLatent(m) "There is not the right number of Z matrices"
    for i = 1:nLatent(m)
        setZ!(m, Z[i], i)
    end
end
