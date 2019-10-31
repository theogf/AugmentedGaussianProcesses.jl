abstract type AbstractGP{T<:Real,L<:Likelihood{T},I<:Inference{T},GP<:Abstract_GP,N} end

const AbstractGP1 = AbstractGP{<:Real,<:Likelihood,<:Inference,<:Abstract_GP,1}

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

get_K(model::AbstractGP1) = model.f[1].K
get_K(model::AbstractGP) = getproperty.(model.f,:K)

get_μ(model::AbstractGP1) = model.f[1].μ
get_μ(model::AbstractGP) = getproperty.(model.f,:μ)

get_Σ(model::AbstractGP1) = model.f[1].Σ
get_Σ(model::AbstractGP) = getproperty.(model.f,:Σ)

get_σ_k(model::AbstractGP1) = model.f[1].σ_k
get_σ_k(model::AbstractGP) = getproperty.(model.f,:σ_k)

get_kernel(model::AbstractGP1) = model.f[1].kernel
get_kernel(model::AbstractGP) = getproperty.(model.f,:kernel)
