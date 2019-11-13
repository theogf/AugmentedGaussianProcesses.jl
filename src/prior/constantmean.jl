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

(μ::ConstantMean{T})(x::Real) where {T<:Real} = μ.C
(μ::ConstantMean{T})(x::AbstractArray) where {T<:Real} = μ.C*ones(T,size(X))
