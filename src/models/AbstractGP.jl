abstract type AbstractGP{T<:Real,L<:Likelihood{T},I<:Inference{T},N} end

const AbstractGP1 = AbstractGP{<:Real,<:Likelihood,<:Inference,1}

@traitdef IsFull{X}
@traitdef IsMultiOutput{X}
@traitdef IsSparse{X}

nLatent(m::AbstractGP) = m.nLatent
nFeatures(m::AbstractGP) = m.nFeatures
nSamples(m::AbstractGP) = m.nSamples
nSamples(m::AbstractGP, i::Int) = m.nSamples[i]

getf(m::AbstractGP) = m.f
getf(m::AbstractGP, i::Int) = getindex(m.f, i)

isTrained(m::AbstractGP) = m.Trained
setTrained!(m::AbstractGP, status::Bool) = m.Trained = status

@traitfn nX(m::TGP) where {TGP<:AbstractGP;!IsMultiOutput{TGP}} = 1
@traitfn nX(m::TGP) where {TGP<:AbstractGP;IsMultiOutput{TGP}} = m.nX

function Random.rand!(model::AbstractGP,A::DenseArray{T},X::AbstractArray{T}) where {T<:Real}
    rand!(MvNormal(predict_f(model,X,covf=true,fullcov=true)...),A)
end

function Random.rand(model::AbstractGP,X::AbstractArray{T},n::Int=1) where {T<:Real}
    if model.nLatent == 1
        rand!(model,Array{T}(undef,size(X,1),n),X)
    else
        @error "Sampling not implemented for multiple output GPs"
    end
end

# Statistics.mean(model::AbstractGP) = model.μ

get_K(model::AbstractGP) = getproperty.(model.f,:K)

get_μ(model::AbstractGP) = getproperty.(model.f,:μ)

get_Σ(model::AbstractGP) = getproperty.(model.f,:Σ)

get_kernel(model::AbstractGP) = getproperty.(model.f,:kernel)


@traitfn function setZ!(m::TGP, Z::AbstractMatrix, i::Int) where {TGP;!IsFull{TGP}}
    @assert size(Z) == size(m.f[i].Z) "Size of Z $(size(Z)) is not the same as in the model $(size(m.f[i].Z))"
    m.f[i].Z.Z = copy(Z)
end

@traitfn function setZ!(m::TGP, Z::AbstractVector{<:AbstractMatrix}) where {TGP; !IsFull{TGP}}
    @assert length(Z) == nLatent(m) "There is not the right number of Z matrices"
    for i in 1:nLatent(m)
        setZ!(m, Z[i], i)
    end
end
