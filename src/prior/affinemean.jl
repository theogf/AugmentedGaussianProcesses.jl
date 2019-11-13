mutable struct AffineMean{T<:Real,V<:AbstractVector{<:Real}} <: PriorMean{T}
    w::V
    b::T
    nDim::Int
    opt::Optimizer
end

"""
**AffineMean**
```julia
    AffineMean(A::V,b::V;opt::Optimizer=Adam(α=0.01))
    AffineMean(dims::Int,features::Int;opt::Optimizer=Adam(α=0.01))
```
Construct an affine operation on `X` : `μ₀(X) = X*w + b` where `w` is a vector and `b` a scalar
Optionally give an optimizer `opt` (`Adam(α=0.01)` by default)
"""
function AffineMean(w::V,b::T,;opt::Optimizer=Adam(α=0.01)) where {V<:AbstractVector{<:Real},T<:Real}
    AffineMean{eltype(w),V}(w,b,length(w),opt)
end

function AffineMean(dims::Int;opt::Optimizer=Adam(α=0.01))
    AffineMean{Float64,Vector{Float64}}(randn(dims),0.0,dims,opt)
end

function update!(μ::AffineMean{T},grad::AbstractVector{T},X::AbstractMatrix) where {T<:Real}
    Δ = vcat(X'*grad,sum(grad))
    Δ = update(μ.opt,Δ)
    μ.w .+= Δ[1:μ.nDim]
    μ.b += Δ[end]
end

function (μ::AffineMean{T})(x::AbstractMatrix) where {T<:Real}
    @assert μ.nDim == size(x,2) "Number of dimensions do not match"
    return x*μ.w .+ μ.b
end
