mutable struct ConstantMean{T<:Real} <: PriorMean{T}
    C::T
    opt::Optimizer
end

"""
**ConstantMean**
```julia
ConstantMean(c::T=1.0;opt::Optimizer=Adam(α=0.01))
```

Construct a prior mean with constant `c`
Optionally set an optimizer `opt` (`Adam(α=0.01)` by default)
"""
function ConstantMean(c::T=1.0;opt::Optimizer=Adam(α=0.01)) where {T<:Real}
    ConstantMean{T}(c,opt)
end

function update!(μ::ConstantMean{T},grad::AbstractVector{T}) where {T<:Real}
    μ.C += update(μ.opt,sum(grad))
end

Base.:+(x::Real,y::ConstantMean{<:Real}) = x+y.C
Base.:+(x::AbstractVector{<:Real},y::ConstantMean{<:Real}) = x.+y.C
Base.:+(x::ConstantMean{<:Real},y::Real) = y+x.C
Base.:+(x::ConstantMean{<:Real},y::AbstractVector{<:Real}) = y.+x.C
Base.:+(x::ConstantMean{<:Real},y::ConstantMean{<:Real}) = ConstantMean(x.C+y.C)
Base.:-(x::Real,y::ConstantMean) = x - y.C
Base.:-(x::AbstractVector{<:Real},y::ConstantMean) = x .- y.C
Base.:-(x::ConstantMean{<:Real},y::Real) = x.C - y
Base.:-(x::ConstantMean{<:Real},y::AbstractVector{<:Real}) = x.C .- y
Base.:-(x::ConstantMean{<:Real},y::ConstantMean{<:Real}) = ConstantMean(x.C-y.C)
Base.:*(A::AbstractMatrix{<:Real},y::ConstantMean{T}) where {T<:Real} = y.C*A*ones(T,size(A,2))
Base.:*(y::ConstantMean{T},A::AbstractMatrix{<:Real}) where {T<:Real} = y.C*ones(T,1,size(A,1))*A
Base.:convert(::T1,x::ConstantMean{T2}) where {T1<:Real,T2<:Real} = T1(x.C)
