struct AffineMean{T<:Real,V<:AbstractVector{<:Real},O} <: PriorMean{T}
    w::V
    b::Base.RefValue{T}
    nDim::Int
    opt::O
end

"""
**AffineMean**
```julia
    AffineMean(A::V,b::V;opt=ADAM(0.01))
    AffineMean(dims::Int,features::Int;opt=ADAM(0.01))
```
Construct an affine operation on `X` : `μ₀(X) = X*w + b` where `w` is a vector and `b` a scalar
Optionally give an optimiser `opt` (`Adam(α=0.01)` by default)
"""
function AffineMean(w::V,b::T,;opt=ADAM(0.01)) where {V<:AbstractVector{<:Real},T<:Real}
    AffineMean{eltype(w),V,typeof(opt)}(w,Ref(b),length(w),opt)
end

function AffineMean(dims::Int;opt=ADAM(0.01))
    AffineMean{Float64,Vector{Float64},typeof(opt)}(randn(dims),Ref(0.0),dims,opt)
end

function update!(opt,μ::AffineMean{T},grad::AbstractVector{T},X::AbstractMatrix) where {T<:Real}
    μ.w .+= Optimise.apply!(opt,μ.w,X'*grad)
    μ.b[] += Optimise.apply!(opt,μ.b,sum(grad))
end

function (μ::AffineMean{T})(x::AbstractMatrix) where {T<:Real}
    @assert μ.nDim == size(x,2) "Number of dimensions of prior weight W ($(size(μ.w))) and X ($(size(x))) do not match"
    return x*μ.w .+ μ.b[]
end
