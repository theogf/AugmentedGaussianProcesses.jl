## File containing different utility functions ##

const OptorNothing = Union{Optimizer,Nothing}

## Jittering object adapting to the type of the GP ##
struct Jittering end;

const jitter = Jittering()

@inline Base.Float64(::Jittering) = Float64(1e-4)
@inline Base.Float32(::Jittering) = Float32(1e-3)
@inline Base.Float16(::Jittering) = Float16(1e-2)
@inline Base.convert(::Type{Float64},::Jittering) = Float64(1e-4)
@inline Base.convert(::Type{Float32},::Jittering) = Float32(1e-3)
@inline Base.convert(::Type{Float16},::Jittering) = Float16(1e-2)

## delta function `(i,j)`, equal `1` if `i == j`, `0` else ##
@inline function δ(i::Integer,j::Integer)
    i == j ? 1.0 : 0.0
end

## Hadamard product between two arrays of same size ##
@inline function hadamard(A::AbstractArray{<:Real},B::AbstractArray{<:Real})
    A.*B
end

## Add on place the transpose of a matrix ##
@inline function add_transpose!(A::AbstractMatrix{<:Real})
    A .+= A'
end

## Return the trace of A*B' ##
@inline function opt_trace(A::AbstractMatrix{<:Real},B::AbstractMatrix{<:Real})
    dot(A,B)
end

## Return the diagonal of A*B' ##
@inline function opt_diag(A::AbstractArray{T,N},B::AbstractArray{T,N}) where {T<:Real,N}
    vec(sum(A.*B,dims=2))
end

## Return the multiplication of Diagonal(v)*B ##
function opt_diag_mul_mat(v::AbstractVector{T},B::AbstractMatrix{T}) where {T<:Real}
    v.*B
end

## Return the multiplication of κᵀ*Diagonal(θ)*κ
@inline function κdiagθκ(κ::AbstractMatrix{T},θ::AbstractVector{T}) where {T<:Real}
    transpose(θ.*κ)*κ
end

## Return the multiplication of ρ*κᵀ*Diagonal(θ)*κ
@inline function ρκdiagθκ(ρ::T,κ::AbstractMatrix{T},θ::AbstractVector{T}) where {T<:Real}
    transpose((ρ*θ).*κ)*κ
end

## Return the addition of a diagonal to a symmetric matrix ##
function opt_add_diag_mat(v::AbstractVector{T},B::AbstractMatrix{T}) where {T<:Real}
    A = copy(B)
    @inbounds for i in 1:size(A,1)
        A[i,i] += v[i]
    end
    A
end

Base.:/(c::AbstractMatrix,a::PDMat) = c*inv(a.chol)

## Temp fix until the deepcopy of the main package is fixed
function Base.copy(opt::Optimizer)
    f = length(fieldnames(typeof(opt)))
    copied_params = [deepcopy(getfield(opt, k)) for k = 1:f]
    return typeof(opt)(copied_params...)
end

## Compute exp(μ)/cosh(c) safely if there is an overflow ##
function safe_expcosh(μ::Real,c::Real)
    return isfinite(exp(μ)/cosh(c)) ? exp(μ)/cosh(c) : 2*logistic(2.0*max(μ,c))
end

## Return a safe version of log(cosh(c)) ##
function logcosh(c::Real)
    return log(exp(-2.0*c)+1.0)+c-logtwo
end


function logisticsoftmax(f::AbstractVector{<:Real})
    return normalize!(logistic.(f),1)
end

function logisticsoftmax(f::AbstractVector{<:Real},i::Integer)
    return logisticsoftmax(f)[i]
end

function symcat(S::Symmetric,v::AbstractVector,vv::Real)
    S = vcat(S,v')
    S = hcat(S,vcat(v,vv))
    return Symmetric(S)
end

export make_grid
function make_grid(range1,range2)
    return hcat([i for i in range1, j in range2][:],[j for i in range1, j in range2][:])
end

function vech(A::AbstractMatrix{T}) where T
    m = size(A,1)
    v = Vector{T}(undef,m*(m+1)÷2)
    k = 0
    for j = 1:m, i = j:m
        @inbounds v[k += 1] = A[i,j]
    end
    return v
end

function duplicate_matrix(n::Int)
    m   = (n * (n + 1)) ÷ 2
    nsq = n^2
    D = zeros(Float64,nsq, m)
    row = 1
    a   = 1
    for i in 1:n
        b = i
        for j = 0:i-2
            D[row + j, b] = 1;
            b = b + n - j - 1;
        end
       row = row + i - 1

       for j = 0:n-i
           D[row + j, a + j] = 1
       end
       row = row + n - i + 1
       a   = a + n - i + 1
    end
    return D
end

function eliminate_matrix(m::Int)
    T = tril(ones(m,m))
    f = findall(T[:].==true)
    k = (m*(m+1))÷2
    m2 = m*m
    L = falses(k,m2)
    x = vec(f .+ m2*(0:(k-1)))
    L[x] .= true
    copy(transpose(L))
end

function commute_transpose(n::Int,m::Int=n)
    d = m*n;
    Tmn = falses(d,d);
    i = 1:d;
    rI = 1 .+ m.*(i.-1)-(m*n-1).*((i.-1).÷n);
    Tmn[[CartesianIndex(v,w) for (w,v) in zip(rI,i)]] .= 1;
    return Tmn
end
