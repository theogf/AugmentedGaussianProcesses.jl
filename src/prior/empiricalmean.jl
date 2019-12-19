mutable struct EmpiricalMean{T<:Real,V<:AbstractVector{<:Real}} <: PriorMean{T}
    C::V
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

function update!(opt,μ::EmpiricalMean{T},grad,X) where {T<:Real}
    μ.C .+= Flux.Optimise.apply!(opt,μ.C,grad)
end

function (μ::EmpiricalMean{T})(x::AbstractMatrix) where {T<:Real}
    @assert size(x,1)==length(μ)
    return μ.C
end
