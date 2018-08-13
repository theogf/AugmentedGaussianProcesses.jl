abstract type Bound{T} end

eltype(::Bound{T}) where T = T

struct OpenBound{T<:Real} <: Bound{T}
    value::T
    function OpenBound{T}(x::Real) where {T<:Real}
        !(T <: Integer) || error("Bounds must be closed for integers")
        if T <: AbstractFloat
            !isnan(x) || error("Bound value must not be NaN")
            !isinf(x) || error("Bound value must not be Inf/-Inf")
        end
        new{T}(x)
    end
end
OpenBound(x::T) where T<:Real = OpenBound{T}(x)

convert(::Type{OpenBound{T}}, b::OpenBound{T}) where T<:Real = b
convert(::Type{OpenBound{T}}, b::OpenBound) where T<:Real = OpenBound{T}(b.value)
string(b::OpenBound) = string("OpenBound(", b.value, ")")


struct ClosedBound{T<:Real} <: Bound{T}
    value::T
    function ClosedBound{T}(x::Real) where {T<:Real}
        if T <: AbstractFloat
            !isnan(x) || error("Bound value must not be NaN")
            !isinf(x) || error("Bound value must not be Inf/-Inf")
        end
        new{T}(x)
    end
end
ClosedBound(x::T) where T<:Real = ClosedBound{T}(x)

convert(::Type{ClosedBound{T}}, b::ClosedBound{T}) where T<:Real = b
convert(::Type{ClosedBound{T}}, b::ClosedBound) where T<:Real = ClosedBound{T}(b.value)
string(b::ClosedBound) = string("ClosedBound(", b.value, ")")


struct NullBound{T<:Real} <: Bound{T} end
NullBound(::Type{T}) where {T<:Real} = NullBound{T}()

convert(::Type{NullBound{T}}, b::NullBound{T}) where T<:Real= b
convert(::Type{NullBound{T}}, b::NullBound) where T<:Real = NullBound{T}()
string(b::NullBound{T}) where T = string("NullBound(", T, ")")


checkvalue(a::NullBound,   x::Real) = true
checkvalue(a::OpenBound,   x::Real) = a.value <  x
checkvalue(a::ClosedBound, x::Real) = a.value <= x

checkvalue(x::Real, a::NullBound)   = true
checkvalue(x::Real, a::OpenBound)   = x <  a.value
checkvalue(x::Real, a::ClosedBound) = x <= a.value

promote_rule(::Type{Bound{T1}},::Type{Bound{T2}}) where {T1,T2} = Bound{promote_rule(T1,T2)}

function show(io::IO, b::T) where T<:Bound
    print(io, string(b))
end
