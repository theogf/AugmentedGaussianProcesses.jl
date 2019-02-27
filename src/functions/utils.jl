"""File containing different utility functions"""

""" Jittering object adapting to the type of the GP """
struct Jittering
function Jittering()
    new()
end
end;

Base.convert(::Type{Float64},::Jittering) = Float64(1e-3)
Base.convert(::Type{Float32},::Jittering) = Float32(1e-2)
Base.convert(::Type{Float16},::Jittering) = Float16(1e-1)

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
function opt_diag(A::AbstractMatrix{<:Real},B::AbstractMatrix{<:Real})
    vec(sum(A.*B,dims=2))
end

# """ Logistic function (1+exp(-x))^-1 """
# function logistic(x::Real)
#     1.0/(1.0+exp(-x))
# end
#
# """ Logistic function (1+exp(-x))^-1 """
# function logistic(x::AbstractVector{<:Real})
#     1.0./(1.0.+exp.(-x))
# end

#Temp fix until the deepcopy of the main package is fixed
function copy(opt::Optimizer)
    f = length(fieldnames(typeof(opt)))
    copied_params = [deepcopy(getfield(opt, k)) for k = 1:f]
    return typeof(opt)(copied_params...)
end

function safe_expcosh(μ::Real,c::Real)
    return isfinite(exp(-0.5*μ)/cosh(0.5*c)) ? exp(-0.5*μ)/cosh(0.5*c) : 2*logistic(max(μ,c))
end

function logcosh(c::Real)
    return log(exp(-2.0*c)+1.0)+c-log(2.0)
end
