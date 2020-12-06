struct AffineMean{T<:Real,V<:AbstractVector{T},O} <: PriorMean{T}
    w::V
    b::Vector{T}
    nDim::Int
    opt::O
end

"""
    AffineMean(A::Vector, b::Real; opt = ADAM(0.01))
    AffineMean(dims::Int, features::Int; opt=ADAM(0.01))

Construct an affine operation on `X` : `μ₀(X) = X*w + b` where `w` is a vector and `b` a scalar
Optionally give an optimiser `opt` (`Adam(α=0.01)` by default)
"""
function AffineMean(
    w::V,
    b::T;
    opt = ADAM(0.01),
) where {T<:Real, V<:AbstractVector{T}}
    AffineMean{T,V,typeof(opt)}(copy(w), [b], length(w), opt)
end

function AffineMean(dims::Int; opt = ADAM(0.01))
    AffineMean{Float64,Vector{Float64},typeof(opt)}(
        randn(Float64, dims),
        [0.0],
        dims,
        opt,
    )
end

Base.show(io::IO, μ₀::AffineMean) = print(
    io,
    "Affine Mean Prior (size(w) = $(length(μ₀.w)), b = $(first(μ₀.b)))",
)

function (μ₀::AffineMean{T})(x::AbstractMatrix) where {T<:Real}
    @assert μ₀.nDim == size(x, 2) "Number of dimensions of prior weight W ($(size(μ₀.w))) and X ($(size(x))) do not match"
    return x * μ₀.w .+ first(μ₀.b)
end

function update!(
    μ₀::AffineMean{T},
    grad,
) where {T<:Real}
    μ₀.w .+= Optimise.apply!(μ₀.opt, μ₀.w, grad.w)
    μ₀.b .+= Optimise.apply!(μ₀.opt, μ₀.b, grad.b)
end
