abstract type AbstractGPModel{T<:Real,L<:AbstractLikelihood,I<:AbstractInference,N} end

@traitdef IsFull{X}
@traitdef IsMultiOutput{X}
@traitdef IsSparse{X}

Base.eltype(::AbstractGPModel{T}) where {T} = T
data(m::AbstractGPModel) = m.data
likelihood(m::AbstractGPModel) = m.likelihood
inference(m::AbstractGPModel) = m.inference
n_latent(::AbstractGPModel{<:Real,<:AbstractLikelihood,<:AbstractInference,N}) where {N} = N
n_output(m::AbstractGPModel) = 1
@traitfn nFeatures(m::TGP) where {TGP <: AbstractGPModel; !IsSparse{TGP}} = nSamples(m)
@traitfn nFeatures(m::TGP) where {TGP <: AbstractGPModel; IsSparse{TGP}} = m.nFeatures

getf(m::AbstractGPModel) = m.f
getf(m::AbstractGPModel, i::Int) = m.f[i]
Base.getindex(m::AbstractGPModel, i::Int) = getf(m, i)

n_iter(m::AbstractGPModel) = n_iter(inference(m))

is_stochastic(m::AbstractGPModel) = is_stochastic(inference(m))

batchsize(m::AbstractGPModel) = batchsize(inference(m))

is_trained(m::AbstractGPModel) = m.trained
set_trained!(m::AbstractGPModel, status::Bool) = m.trained = status

verbose(m::AbstractGPModel) = m.verbose

post_step!(::AbstractGPModel, ::Any) = nothing

function Random.rand!(
    model::AbstractGPModel, A::DenseArray{T}, X::AbstractVector
) where {T<:Real}
    return rand!(MvNormal(predict_f(model, X; cov=true, diag=false)...), A)
end

function Random.rand(model::AbstractGPModel, X::AbstractMatrix, n::Int=1)
    return rand(model, KernelFunctions.RowVecs(X), n)
end

function Random.rand(model::AbstractGPModel{T}, X::AbstractVector, n::Int=1) where {T<:Real}
    if n_latent(model) == 1
        rand!(model, Array{T}(undef, length(X), n), X)
    else
        @error "Sampling not implemented for multiple output GPs"
    end
end

pr_covs(model::AbstractGPModel, x) = pr_cov.(model.f, Ref(x))

pr_means(model::AbstractGPModel) = pr_mean.(model.f)
pr_means(model::AbstractGPModel, X::AbstractVector) = pr_mean.(model.f, Ref(X))

setpr_means!(model::AbstractGPModel, pr_means) = setpr_mean!.(model.f, pr_means)

means(model::AbstractGPModel) = mean.(model.f)

covs(model::AbstractGPModel) = cov.(model.f)

kernels(model::AbstractGPModel) = kernel.(model.f)

setkernels!(model::AbstractGPModel, kernels) = setkernel!.(model.f, kernels)

setZs!(model::AbstractGPModel, Zs) = setZ!.(model.f, Zs)
