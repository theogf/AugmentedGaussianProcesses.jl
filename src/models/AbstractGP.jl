abstract type AbstractGP{T<:Real,L<:Likelihood{T},I<:Inference{T},V<:AbstractArray{T}} end

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

covariance(model::AbstractGP) = model.Σ

diag_covariance(model::AbstractGP) = diag.(model.Σ)

prior_mean(model::AbstractGP) = model.μ₀

Base.length(model::AbstractGP) = model.nLatent

kernel(model::AbstractGP) = model.kernel
