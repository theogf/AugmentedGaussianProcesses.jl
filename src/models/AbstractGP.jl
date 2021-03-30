abstract type AbstractGP{T<:Real,L<:AbstractLikelihood{T},I<:AbstractInference{T},N} end

@traitdef IsFull{X}
@traitdef IsMultiOutput{X}
@traitdef IsSparse{X}

Base.eltype(::AbstractGP{T}) where {T} = T
data(m::AbstractGP) = m.data
likelihood(m::AbstractGP) = m.likelihood
inference(m::AbstractGP) = m.inference
nLatent(::AbstractGP{<:Real,<:AbstractLikelihood,<:AbstractInference, N}) where {N} = N
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

post_step!(::AbstractGP) = nothing

function Random.rand!(
    model::AbstractGP,
    A::DenseArray{T},
    X::AbstractVector,
) where {T<:Real}
    rand!(MvNormal(predict_f(model, X, cov = true, diag = false)...), A)
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

pr_covs(model::AbstractGP) = pr_cov.(model.f)

pr_means(model::AbstractGP) = pr_mean.(model.f)
pr_means(model::AbstractGP, X::AbstractVector) = pr_mean.(model.f, Ref(X))

setpr_means!(model::AbstractGP, pr_means) = setpr_mean!.(model.f, pr_means)

means(model::AbstractGP) = mean.(model.f)

covs(model::AbstractGP) = cov.(model.f)

kernels(model::AbstractGP) = kernel.(model.f)

setkernels!(model::AbstractGP, kernels) = setkernel!.(model.f, kernels)

setZs!(model::AbstractGP, Zs) = setZ!.(model.f, Zs)
