"""
Make prediction for the next `t` times. Assumes that `y_past` is already ordered.

"""
@traitfn function predict_ar(m::TGP, p::Int = 3, n::Int = 1; y_past = get_y(m)) where {T,TGP<:AbstractGP{T};!IsMultiOutput{TGP}}
    @assert y_past isa AbstractVector{<:Real}
    @assert length(y_past) >= p
    Xtest = reshape(y_past[end-p:end],1,:)
    y_new = zeros(T, n)
    for i in 1:n
        y_new[i] = _predict_f(m, Xtest, covf = false)
        Xtest = hcat(Xtest[:,2:end],y_new[i])
    end
    return y_new
end

@traitfn function predict_ar(m::TGP, p::Int = 3, n::Int = 1; y_past = get_y(m)) where {T,TGP<:AbstractGP{T};IsMultiOutput{TGP}}
    @assert y_past isa AbstractVector{<:AbstractVector}
    @assert length(y_past) == m.nTask
    @assert all(length.(y_past).>= p)
    Xtest = [reshape(y[end-p:end],1,:) for y in y_past]
    y_new = [zeros(T, n) for _ in 1:m.nTask]
    for i in 1:n
        setindex!.(y_new, i, _predict_f(m, Xtest, covf = false))
        Xtest = hcat.(getindex.(Xtest, 1, Ref(2:p)), getindex.(y_new, i))
    end
    return y_new
end

@traitfn function sample_ar(m::TGP, p::Int, n::Int = 1; y_past = get_y(m)) where {T,TGP<:AbstractGP{T};!IsMultiOutput{TGP}}
    @assert y_past isa AbstractVector{<:Real}
    @assert length(y_past) >= p
    Xtest = reshape(y_past[end-p:end],1,:)
    y_new = zeros(T, n)
    for i in 1:n
        μ, σ² = _predict_f(m, Xtest, covf = false)
        y_new[i] = rand(Normal(first(μ), sqrt(first(σ²))))
        Xtest = hcat(Xtest[:,2:end],y_new[i])
    end
    return y_new
end

@traitfn function sample_ar(m::TGP, p::Int, n::Int = 1; y_past = get_y(m)) where {T,TGP<:AbstractGP{T};IsMultiOutput{TGP}}
    @assert y_past isa AbstractVector{<:AbstractVector}
    @assert length(y_past) == m.nTask
    @assert all(length.(y_past).>= p)
    Xtest = [reshape(y[end-p:end],1,:) for y in y_past]
    y_new = [zeros(T, n) for _ in 1:m.nTask]
    for i in 1:n
        μ, σ² = _predict_f(m, Xtest, covf = false)
        setindex!.(y_new, i, rand.(Normal.(first.(μ), sqrt.(first.(σ²)))))
        Xtest = hcat.(getindex.(Xtest, 1, Ref(2:p)), getindex.(y_new, i))
    end
    return y_new
end
