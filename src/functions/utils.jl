"""File containing different utility functions"""

struct Jittering
function Jittering()
    new()
end
end;

Base.convert(::Type{Float64},::Jittering) = Float64(1e-3)
Base.convert(::Type{Float32},::Jittering) = Float32(1e-2)
Base.convert(::Type{Float16},::Jittering) = Float16(1e-1)

function δ(i::Integer,j::Integer)
    i == j ? 1.0 : 0.0
end

"""Hadamard product between two arrays of same size"""
function hadamard(A::AbstractArray{<:Real},B::AbstractArray{<:Real})
    A.*B
end


function logisticsoftmax(f::AbstractVector{<:Real})
    s = logit.(f)
    return s./sum(s)
end

function logisticsoftmax(f::AbstractVector{<:Real},i::Integer)
    return logisticsoftmax(f)[i]
end

function grad_logisticsoftmax(s::AbstractVector{<:Real},σ::AbstractVector{<:Real},i::Integer)
    base_grad = -s.*(1.0.-σ).*s[i]
    base_grad[i] += s[i]*(1.0-σ[i])
    return base_grad
end

function hessian_logisticsoftmax(s::AbstractVector{<:Real},σ::AbstractVector{<:Real},i::Integer)
    m = length(s)
    hessian = zeros(m,m)
    for j in 1:m
        for k in 1:m
            hessian[j,k] = (1-σ[j])*s[i]*(
            (δ(i,k)-s[k])*(1.0-σ[k])*(δ(i,j)-s[j])
            -s[j]*(δ(j,k)-s[k])*(1.0-σ[k])
            -δ(k,j)*σ[j]*(δ(i,j)-s[j]))
        end
    end
    return hessian
end


function add_transpose!(A::AbstractMatrix{<:Real})
    A .+= A'
end

function softmax(f::AbstractVector{<:Real})
    s = exp.(f)
    return s./sum(s)
end

function softmax(f::AbstractVector{<:Real},i::Integer)
    return softmax(f)[i]
end

function grad_softmax(s::AbstractVector{<:Real},i::Integer)
    base_grad = -s.*s[i]
    base_grad[i] += s[i]
    return base_grad
end

function hessian_softmax(s::AbstractVector{<:Real},i::Integer)
    m = length(s)
    hessian = zeros(m,m)
    for j in 1:m
        for k in 1:m
            hessian[j,k] = s[i]*((δ(i,k)-s[k])*(δ(i,j)-s[j])-s[j]*(δ(j,k)-s[k]))
        end
    end
    return hessian
end
