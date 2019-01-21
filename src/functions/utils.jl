
function δ(i::Integer,j::Integer)
    i == j ? 1.0 : 0.0
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


function add_transpose!(A::AbstractMatrix{T}) where {T}
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
