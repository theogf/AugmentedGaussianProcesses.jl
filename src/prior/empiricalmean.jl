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

Base.:+(x::Real,y::EmpiricalMean{<:Real}) = x.+y.C
Base.:+(x::AbstractVector{<:Real},y::EmpiricalMean{<:Real}) = x+y.C
Base.:+(x::EmpiricalMean{<:Real},y::Real) = y.+x.C
Base.:+(x::EmpiricalMean{<:Real},y::AbstractVector{<:Real}) = y+x.C
Base.:+(x::EmpiricalMean{<:Real},y::EmpiricalMean{<:Real}) = EmpiricalMean(x.C+y.C)
Base.:-(x::Real,y::EmpiricalMean) = x .- y.C
Base.:-(x::AbstractVector{<:Real},y::EmpiricalMean) = x - y.C
Base.:-(x::EmpiricalMean{<:Real},y::Real) = x.C .- y
Base.:-(x::EmpiricalMean{<:Real},y::AbstractVector{<:Real}) = x.C - y
Base.:-(x::EmpiricalMean{<:Real},y::EmpiricalMean{<:Real}) = EmpiricalMean(x.C-y.C)
Base.:*(A::AbstractMatrix{<:Real},y::EmpiricalMean{T}) where {T<:Real} = A*y.C
Base.:*(y::EmpiricalMean{T},A::AbstractMatrix{<:Real}) where {T<:Real} = transpose(y)*A
Base.:convert(::T1,x::EmpiricalMean{T2}) where {T1<:Real,T2<:Real} = T1(x.C)
