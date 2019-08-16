mutable struct ZeroMean{T<:Real} <: PriorMean{T}
end

"""
ZeroMean
```julia
ZeroMean()
```
Construct a mean prior set to 0 and cannot be changed.
"""
function ZeroMean()
    ZeroMean{Float64}()
end

function update!(μ::ZeroMean{T},grad::AbstractVector{T}) where {T<:Real}
end

array(μ::ZeroMean{T},length::Int) where {T<:Real} = zeros(T,length)

Base.:+(x::Real,y::ZeroMean{<:Real}) = x
Base.:+(x::AbstractVector{<:Real},y::ZeroMean{<:Real}) = x
Base.:+(x::ZeroMean{<:Real},y::Real) = y
Base.:+(x::ZeroMean{<:Real},y::AbstractVector{<:Real}) = y
Base.:+(x::ZeroMean{<:Real},y::ConstantMean{<:Real}) = ConstantMean(y.C)
Base.:-(x::Real,y::ZeroMean{<:Real}) = x
Base.:-(x::AbstractVector{<:Real},y::ZeroMean) = x
Base.:-(x::ZeroMean{<:Real},y::Real) = -y
Base.:-(x::ZeroMean{<:Real},y::AbstractVector{<:Real}) = -y
Base.:-(x::ZeroMean{<:Real},y::ConstantMean{<:Real}) = ConstantMean(-y.C)
Base.:*(A::AbstractMatrix{<:Real},y::ZeroMean{T}) where {T<:Real} = zeros(T,size(A,2))
Base.:*(y::ZeroMean{T},A::AbstractMatrix{<:Real}) where {T<:Real} = zeros(T,1,size(A,1))
Base.:convert(::T1,x::ZeroMean{T2}) where {T1<:Real,T2<:Real} = T1(x.C)
