struct AffineMean{T<:Real,V<:AbstractVector{T},O} <: PriorMean{T}
    w::V
    b::Vector{T}
    nDim::Int
    opt::O
end

"""
    AffineMean(w::Vector, b::Real; opt = ADAM(0.01))
    AffineMean(dims::Int; opt=ADAM(0.01))

## Arguments
- `w::Vector` : Weight vector
- `b::Real` : Bias
- `dims::Int` : Number of features per vector

Construct an affine operation on `X` : `μ₀(X) = X * w + b` where `w` is a vector and `b` a scalar
Optionally give an optimiser `opt` (`Adam(α=0.01)` by default)
"""
function AffineMean(w::V, b::T; opt=ADAM(0.01)) where {T<:Real,V<:AbstractVector{T}}
    return AffineMean{T,V,typeof(opt)}(copy(w), [b], length(w), opt)
end

function AffineMean(dims::Int; opt=ADAM(0.01))
    return AffineMean{Float64,Vector{Float64},typeof(opt)}(
        randn(Float64, dims), [0.0], dims, opt
    )
end

function Base.show(io::IO, ::MIME"text/plain", μ₀::AffineMean)
    return print(
        io, "Affine Mean Prior (size(w) = ", length(μ₀.w), ", b = ", only(μ₀.b), ")"
    )
end

function (μ₀::AffineMean{T})(x::AbstractVector) where {T<:Real}
    # μ₀.nDim == size(x, 1) || error(
    # "Number of dimensions of prior weight W (",
    # size(μ₀.w),
    # ") and X (",
    # size(x),
    # ") do not match",
    # )
    return dot.(x, Ref(μ₀.w)) .+ only(μ₀.b)
end

function init_priormean_state(hyperopt_state, μ₀::AffineMean)
    μ₀_state = (; w=Optimisers.init(μ₀.opt, μ₀.w), b=Optimisers.init(μ₀.opt, μ₀.b))
    return merge(hyperopt_state, (; μ₀_state))
end

function update!(μ₀::AffineMean{T}, hyperopt_state, grad) where {T<:Real}
    μ₀_state = hyperopt_state.μ₀_state
    w, Δw = Optimisers.apply(μ₀.opt, μ₀_state.w, μ₀.w, grad.w)
    b, Δb = Optimisers.apply(μ₀.opt, μ₀_state.b, μ₀.b, grad.b)
    μ₀.w .+= Δw
    μ₀.b .+= Δb
    return merge(hyperopt_state, (; μ₀_state=(; w, b)))
end
