"""
Make prediction for the next `t` times. Assumes that `y_past` is already ordered.

"""
@traitfn function predict_ar(
    m::TGP, p::Int=3, n::Int=1; y_past=get_y(m)
) where {T,TGP<:AbstractGPModel{T};!IsMultiOutput{TGP}}
    @assert y_past isa AbstractVector{<:Real}
    @assert length(y_past) >= p
    Xtest = reshape(y_past[(end - p + 1):end], 1, :)
    y_new = zeros(T, n)
    for i in 1:n
        y_new[i] = first(first(first(_predict_f(m, Xtest; cov=false))))
        Xtest = hcat(Xtest[:, 2:end], y_new[i])
    end
    return y_new
end

ks = []

@traitfn function predict_ar(
    m::TGP, p::Int=3, n::Int=1; y_past=get_y(m)
) where {T,TGP<:AbstractGPModel{T};IsMultiOutput{TGP}}
    @assert y_past isa AbstractVector{<:AbstractVector}
    @assert length(y_past) == m.nTask
    @assert all(length.(y_past) .>= p)
    @assert all(m.nDim .== p)
    Xtest = [reshape(y[(end - p + 1):end], 1, :) for y in y_past]
    y_new = [zeros(T, n) for _ in 1:(m.nTask)]
    for i in 1:n
        setindex!.(y_new, first.(first.(first(_predict_f(m, Xtest; cov=false)))), i)
        Xtest = [hcat(Xtest[j][:, 2:end], y_new[j][i]) for j in 1:(m.nTask)]
    end
    return y_new
end

@traitfn function sample_ar(
    m::TGP, p::Int, n::Int=1; y_past=get_y(m)
) where {T,TGP<:AbstractGPModel{T};!IsMultiOutput{TGP}}
    @assert y_past isa AbstractVector{<:Real}
    @assert length(y_past) >= p
    Xtest = reshape(y_past[(end - p):end], 1, :)
    y_new = zeros(T, n)
    for i in 1:n
        μ, σ² = _predict_f(m, Xtest; cov=true)
        y_new[i] = rand(Normal(first(μ), sqrt(first(σ²))))
        Xtest = hcat(Xtest[:, 2:end], y_new[i])
    end
    return y_new
end

@traitfn function sample_ar(
    m::TGP, p::Int, n::Int=1; y_past=get_y(m)
) where {T,TGP<:AbstractGPModel{T};IsMultiOutput{TGP}}
    @assert y_past isa AbstractVector{<:AbstractVector}
    @assert length(y_past) == m.nTask
    @assert all(length.(y_past) .>= p)
    Xtest = [reshape(y[(end - p + 1):end], 1, :) for y in y_past]
    y_new = [zeros(T, n) for _ in 1:(m.nTask)]
    for i in 1:n
        μ, σ² = _predict_f(m, Xtest; cov=true)
        μ = first.(first.(μ))
        σ² = first.(first.(σ²))
        setindex!.(y_new, rand.(Normal.(μ, sqrt.(σ²))), i)
        Xtest = [hcat(Xtest[j][:, 2:end], y_new[j][i]) for j in 1:(m.nTask)]
    end
    return y_new
end
