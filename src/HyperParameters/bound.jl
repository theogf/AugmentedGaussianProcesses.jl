abstract type Bound{T} end

eltype{T}(::Bound{T}) = T

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
OpenBound{T<:Real}(x::T) = OpenBound{T}(x)

convert{T<:Real}(::Type{OpenBound{T}}, b::OpenBound{T}) = b
convert{T<:Real}(::Type{OpenBound{T}}, b::OpenBound) = OpenBound{T}(b.value)
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
ClosedBound{T<:Real}(x::T) = ClosedBound{T}(x)

convert{T<:Real}(::Type{ClosedBound{T}}, b::ClosedBound{T}) = b
convert{T<:Real}(::Type{ClosedBound{T}}, b::ClosedBound) = ClosedBound{T}(b.value)
string(b::ClosedBound) = string("ClosedBound(", b.value, ")")


struct NullBound{T<:Real} <: Bound{T} end
NullBound{T<:Real}(::Type{T}) = NullBound{T}()

convert{T<:Real}(::Type{NullBound{T}}, b::NullBound{T}) = b
convert{T<:Real}(::Type{NullBound{T}}, b::NullBound) = NullBound{T}()
string{T}(b::NullBound{T}) = string("NullBound(", T, ")")


checkvalue(a::NullBound,   x::Real) = true
checkvalue(a::OpenBound,   x::Real) = a.value <  x
checkvalue(a::ClosedBound, x::Real) = a.value <= x

checkvalue(x::Real, a::NullBound)   = true
checkvalue(x::Real, a::OpenBound)   = x <  a.value
checkvalue(x::Real, a::ClosedBound) = x <= a.value

promote_rule{T1,T2}(::Type{Bound{T1}},::Type{Bound{T2}}) = Bound{promote_rule(T1,T2)}

function show{T<:Bound}(io::IO, b::T)
    print(io, string(b))
end
