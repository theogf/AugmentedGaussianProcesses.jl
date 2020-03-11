struct ZeroMean{T<:Real} <: PriorMean{T} end

"""
ZeroMean
```julia
ZeroMean()
```
Construct a mean prior set to 0 and which cannot be updated.
"""
function ZeroMean()
    ZeroMean{Float64}()
end

update!(μ::ZeroMean{T},grad::AbstractVector{T},x) where {T<:Real} = nothing

(μ::ZeroMean{T})(x::Real) where {T<:Real} = zero(T)
(μ::ZeroMean{T})(x::AbstractMatrix) where {T<:Real} = zeros(T,size(x,1))
