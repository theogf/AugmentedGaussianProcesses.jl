## File containing different utility functions ##

## Jittering object adapting to the type of the GP ##
struct Jittering end;

const jitt = Jittering()

@inline Base.Float64(::Jittering) = Float64(1e-4)
@inline Base.Float32(::Jittering) = Float32(1e-3)
@inline Base.Float16(::Jittering) = Float16(1e-2)
@inline Base.convert(::Type{Float64}, ::Jittering) = Float64(1e-4)
@inline Base.convert(::Type{Float32}, ::Jittering) = Float32(1e-3)
@inline Base.convert(::Type{Float16}, ::Jittering) = Float16(1e-2)

# Compute expectation given the nodes
function expectation(f, μ::Real, σ²::Real)
    x = pred_nodes * sqrt(max(σ², zero(σ²))) .+ μ
    return dot(pred_weights, f.(x))
end

# Return √E[f^2]
function sqrt_expec_square(μ, σ²)
    return sqrt(abs2(μ) + σ²)
end

# Return √E[(f-y)^2]
function sqrt_expec_square(μ, σ², y)
    return sqrt(abs2(μ - y) + σ²)
end

## delta function `(i,j)`, equal `1` if `i == j`, `0` else ##
@inline function δ(T, i::Integer, j::Integer)
    return ifelse(i == j, one(T), zero(T))
end
δ(i::Integer, j::Integer) = δ(Float64, i, j)

## Hadamard product between two arrays of same size ##
@inline function hadamard(A::AbstractArray{<:Real}, B::AbstractArray{<:Real})
    return A .* B
end

## Add on place the transpose of aZero Mean Prior matrix ##
@inline function add_transpose!(A::AbstractMatrix{<:Real})
    return A .+= A'
end

invquad(a::Cholesky, x::AbstractVector) = sum(abs2, a.L \ x)

## Return the trace of A*B' ##
@inline function trace_ABt(A::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real})
    return dot(A, B)
end

## Return the diagonal of A*B' ##
@inline function diag_ABt(A::AbstractMatrix, B::AbstractMatrix)
    return vec(sum(A .* B; dims=2))
end

## Return the multiplication of Diagonal(v)*B ##
function diagv_B(v::AbstractVector, B::AbstractMatrix)
    return v .* B
end

## Return the multiplication of κᵀ*Diagonal(θ)*κ
@inline function κdiagθκ(κ::AbstractMatrix{T}, θ::AbstractVector{T}) where {T<:Real}
    return transpose(θ .* κ) * κ
end

## Return the multiplication of ρ*κᵀ*Diagonal(θ)*κ
@inline function ρκdiagθκ(ρ::T, κ::AbstractMatrix{T}, θ::AbstractVector{T}) where {T<:Real}
    return transpose((ρ * θ) .* κ) * κ
end

## Return the addition of a diagonal to a symmetric matrix ##
function opt_add_diag_mat(v::AbstractVector{T}, B::AbstractMatrix{T}) where {T<:Real}
    A = copy(B)
    @inbounds for i in 1:size(A, 1)
        A[i, i] += v[i]
    end
    return A
end

## Compute exp(μ)/cosh(c) safely if there is an overflow ##
function safe_expcosh(μ::Real, c::Real)
    return isfinite(exp(μ) / cosh(c)) ? exp(μ) / cosh(c) : 2 * logistic(2.0 * max(μ, c))
end

## Return a safe version of log(cosh(c)) ##
function logcosh(c::Real)
    return log(exp(-2.0 * c) + 1.0) + c - logtwo
end

function symcat(S::Symmetric, v::AbstractVector, vv::Real)
    S = vcat(S, v')
    S = hcat(S, vcat(v, vv))
    return Symmetric(S)
end

export make_grid
function make_grid(range1, range2)
    return hcat([i for i in range1, j in range2][:], [j for i in range1, j in range2][:])
end

Base.:*(x::Real, C::Cholesky) = Cholesky(sqrt(x) * C.factors, C.uplo, C.info)

Base.:*(C::Cholesky, x::AbstractVecOrMat) = C.L * (transpose(C.L) * x)

Base.:+(C::Cholesky, x::UniformScaling) = cholesky(Symmetric(Array(C) + x))
