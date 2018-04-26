immutable Interval{T<:Real,A<:Bound{T},B<:Bound{T}}
    a::A
    b::B
    function Interval{T}(a::A, b::B) where {T<:Real,A<:Bound{T},B<:Bound{T}}
        if !(A <: NullBound || B <: NullBound)
            va = a.value
            vb = b.value
            if A <: ClosedBound && B <: ClosedBound
                va <= vb || error("Invalid bounds: a=$va must be less than or equal to b=$vb")
            else
                va < vb || error("Invalid bounds: a=$va must be less than b=$vb")
            end
        end
        new{T,A,B}(a,b)
    end
end
Interval(a::Bound{T}, b::Bound{T}) where {T<:Real} = Interval{T}(a,b)

eltype{T}(::Interval{T}) = T

interval(a::Void, b::Void) = Interval(NullBound{Float64}(), NullBound{Float64}())
interval(a::Bound{T}, b::Void) where {T<:Real} = Interval(a, NullBound{T}())
interval(a::Void, b::Bound{T}) where {T<:Real} = Interval(NullBound{T}(), b)
interval(a::Bound{T}, b::Bound{T}) where {T<:Real} = Interval(a,b)
interval(::Type{T}) where {T<:Real} = Interval(NullBound{T}(), NullBound{T}())


checkvalue(I::Interval, x::Real) = checkvalue(I.a, x) && checkvalue(x, I.b)

function theta{T<:AbstractFloat,A,B}(I::Interval{T,A,B}, x::T)
    checkvalue(I,x) || throw(DomainError())
    if A <: OpenBound
        return B <: OpenBound ? log(x-I.a.value) - log(I.b.value-x) : log(x-I.a.value)
    else
        return B <: OpenBound ? log(I.b.value-x) : x 
    end
end

function upperboundtheta{T<:AbstractFloat,A,B}(I::Interval{T,A,B})
    if B <: ClosedBound
        return A <: OpenBound ? log(I.b.value - I.a.value) : I.b.value
    elseif B <: OpenBound
        return A <: ClosedBound ? log(I.b.value - I.a.value) : convert(T,Inf)
    else
        return convert(T,Inf)
    end
end

function lowerboundtheta{T<:AbstractFloat,A,B}(I::Interval{T,A,B})
    A <: ClosedBound && !(B <: OpenBound) ? I.a.value : convert(T,-Inf)
end

function checktheta{T<:AbstractFloat}(I::Interval{T}, x::T)
    lowerboundtheta(I) <= x <= upperboundtheta(I)
end

function eta{T<:AbstractFloat,A,B}(I::Interval{T,A,B}, x::T)
    checktheta(I,x) || throw(DomainError())
    if A <: OpenBound
        if B <: OpenBound
            return (I.b.value*exp(x) + I.a.value)/(one(T) + exp(x))
        else
            return exp(x) + I.a.value
        end
    else
        return B <: OpenBound ? I.b.value - exp(x) : x 
    end
end

function string{T1,T2,T3}(I::Interval{T1,T2,T3})
    if T2 <: NullBound
        if T3 <: NullBound
            string("interval(", T1, ")")
        else
            string("interval(nothing,", string(I.b), ")")
        end
    else
        string("interval(", string(I.a), ",", T3 <: NullBound ? "nothing" : string(I.b), ")")
    end
end

function show(io::IO, I::Interval)
    print(io, string(I))
end
