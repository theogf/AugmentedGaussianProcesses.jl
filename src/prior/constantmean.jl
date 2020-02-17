mutable struct ConstantMean{T<:Real,O} <: PriorMean{T}
    C::Base.RefValue{T}
    opt::O
end

"""
**ConstantMean**
```julia
ConstantMean(c::T=1.0;opt=ADMA(0.01))
```

Construct a prior mean with constant `c`
Optionally set an optimiser `opt` (`ADAM(0.01)` by default)
"""
function ConstantMean(c::T=1.0;opt=ADAM(0.01)) where {T<:Real}
    ConstantMean{T,typeof(opt)}(Ref(c),opt)
end

function update!(opt,μ::ConstantMean{T},grad::AbstractVector{T},X) where {T<:Real}
    μ.C[] += Optimise.apply!(opt,μ.C,sum(grad))
end

(μ::ConstantMean{T})(x::Real) where {T<:Real} = μ.C[]
(μ::ConstantMean{T})(x::AbstractMatrix) where {T<:Real} = fill(T(μ.C[]),size(X,1))
