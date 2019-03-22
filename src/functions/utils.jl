"""File containing different utility functions"""

""" Jittering object adapting to the type of the GP """
struct Jittering
function Jittering()
    new()
end
end;

Base.convert(::Type{Float64},::Jittering) = Float64(1e-5)
Base.convert(::Type{Float32},::Jittering) = Float32(1e-4)
Base.convert(::Type{Float16},::Jittering) = Float16(1e-3)

""" delta function `(i,j)`, equal `1` if `i == j`, `0` else """
@inline function δ(i::Integer,j::Integer)
    i == j ? 1.0 : 0.0
end

"""Hadamard product between two arrays of same size"""
function hadamard(A::AbstractArray{<:Real},B::AbstractArray{<:Real})
    A.*B
end

""" Add on place the transpose of a matrix """
function add_transpose!(A::AbstractMatrix{<:Real})
    A .+= A'
end

""" Return the trace of A*B' """
function opt_trace(A::AbstractMatrix{<:Real},B::AbstractMatrix{<:Real})
    dot(A,B)
end

""" Return the diagonal of A*B' """
function opt_diag(A::AbstractArray{T,N},B::AbstractArray{T,N}) where {T<:Real,N}
    vec(sum(A.*B,dims=2))
end

""" Return the multiplication of Diagonal(v)*B """
function opt_diag_mul_mat(v::AbstractVector{T},B::AbstractMatrix{T}) where {T<:Real}
    v.*B
end

""" Return the addition of a diagonal to a symmetric matrix """
function opt_add_diag_mat(v::AbstractVector{T},B::AbstractMatrix{T}) where {T<:Real}
    A = copy(B)
    @inbounds for i in 1:size(A,1)
        A[i,i] += v[i]
    end
    A
end

#Temp fix until the deepcopy of the main package is fixed
function copy(opt::Optimizer)
    f = length(fieldnames(typeof(opt)))
    copied_params = [deepcopy(getfield(opt, k)) for k = 1:f]
    return typeof(opt)(copied_params...)
end

"""Compute exp(μ)/cosh(c) safely if there is an overflow"""
function safe_expcosh(μ::Real,c::Real)
    return isfinite(exp(μ)/cosh(c)) ? exp(μ)/cosh(c) : 2*logistic(2.0*max(μ,c))
end

"""Return a safe version of log(cosh(c))"""
function logcosh(c::Real)
    return log(exp(-2.0*c)+1.0)+c-logtwo
end


function logisticsoftmax(f::AbstractVector{<:Real})
    s = logistic.(f)
    return s/sum(s)
end

function logisticsoftmax(f::AbstractVector{<:Real},i::Integer)
    return logisticsoftmax(f)[i]
end
