abstract type AbstractGP{L<:Likelihood,I<:Inference,T<:Real,V<:AbstractArray{T}} end

abstract type SparseGP{L<:Likelihood,I<:Inference,T<:Real,V<:AbstractArray{T}} <: AbstractGP{L,I,T,V} end

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
