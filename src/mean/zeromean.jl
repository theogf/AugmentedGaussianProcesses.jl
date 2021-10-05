struct ZeroMean{T<:Real} <: PriorMean{T} end

"""
    ZeroMean()

Construct a mean prior set to `0` and which cannot be updated.
"""
function ZeroMean()
    return ZeroMean{Float64}()
end

Base.show(io::IO, ::MIME"text/plain", ::ZeroMean) = print(io, "Zero Mean Prior")

update!(::ZeroMean{T}, ::Any, ::Any) where {T<:Real} = nothing

(μ::ZeroMean{T})(::Real) where {T<:Real} = zero(T)
(μ::ZeroMean{T})(x::AbstractVector) where {T} = zeros(T, length(x))
