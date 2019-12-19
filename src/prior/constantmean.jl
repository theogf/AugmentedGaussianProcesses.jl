mutable struct ConstantMean{T<:Real} <: PriorMean{T}
    C::Ref{T}
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

function update!(opt,μ::ConstantMean{T},grad::AbstractVector{T},X) where {T<:Real}
    μ.C[] += Flux.Optimise.apply!(opt,μ.C,sum(grad))
end

(μ::ConstantMean{T})(x::Real) where {T<:Real} = μ.C[]
(μ::ConstantMean{T})(x::AbstractMatrix) where {T<:Real} = fill(T(μ.C[]),size(X,1))
