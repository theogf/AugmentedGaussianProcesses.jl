mutable struct EmpiricalMean{T<:Real,V<:AbstractVector{<:Real}} <: PriorMean{T}
    C::V
    opt::Optimizer
end

"""
**EmpiricalMean**
```julia`
function EmpiricalMean(c::V=1.0;opt::Optimizer=Adam(α=0.01)) where {V<:AbstractVector{<:Real}}
```
Construct a constant mean with values `c`
Optionally give an optimizer `opt` (`Adam(α=0.01)` by default)
"""
function EmpiricalMean(c::V=1.0;opt::Optimizer=Adam(α=0.01)) where {V<:AbstractVector{<:Real}}
    EmpiricalMean{eltype(c),V}(c,opt)
end

function update!(μ::EmpiricalMean{T},grad::AbstractVector{T}) where {T<:Real}
    μ.C .+= update!(μ.opt,grad)
end

function (μ::EmpiricalMean{T})(x::AbstractMatrix) where {T<:Real}
    @assert size(x,1)==length(μ)
    return μ.C
end
