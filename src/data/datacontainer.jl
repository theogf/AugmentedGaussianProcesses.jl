abstract type AbstractDataContainer end

n_sample(d::AbstractDataContainer) = d.n_sample
n_dim(d::AbstractDataContainer) = d.n_dim
n_output(d::AbstractDataContainer) = 1

input(d::AbstractDataContainer) = d.X
output(d::AbstractDataContainer) = d.y

struct DataContainer{Tx<:Real,TX<:AbstractVector,Ty<:Real,TY<:AbstractArray} <:
       AbstractDataContainer
    X::TX # Feature vectors
    y::TY # Output (-1,1 for classification, real for regression, bitmatrix for multiclass)
    n_sample::Int # Number of samples
    n_dim::Int # Number of features per sample
end

function wrap_data(X::TX, y::TY) where {TX,TY<:AbstractMatrix}
    size(y, 1) == length(X) || error(
        "There is not the same number of samples in X ($(length(X))) and y ($(size(y, 1)))",
    )
    Tx = eltype(first(X))
    Ty = eltype(first(y))
    return DataContainer{Tx,TX,Ty,TY}(X, y, length(X), length(first(X)))
end

function wrap_data(X::TX, y::TY) where {TX,TY<:AbstractVector{<:Real}}
    length(y) == length(X) || error(
        "There is not the same number of samples in X ($(length(X))) and y ($(length(y)))",
    )
    Tx = eltype(first(X))
    Ty = eltype(first(y))
    return DataContainer{Tx,TX,Ty,TY}(X, y, length(X), length(first(X)))
end

function wrap_data(X::TX, y::TY) where {TX,TY<:AbstractVector}
    all(length.(y) .== length(X)) || error(
        "There is not the same number of samples in X ($(length(X))) and y ($(length.(y))))",
    )
    Tx = eltype(first(X))
    Ty = eltype(first(y))
    return DataContainer{Tx,TX,Ty,TY}(X, y, length(X), length(first(X)))
end

struct MODataContainer{Tx<:Real,TX<:AbstractVector,TY<:AbstractVector} <:
       AbstractDataContainer
    X::TX # Feature vectors
    y::TY # Output ({-1,1} for classification, real for regression, matrix for multiclass)
    n_sample::Int # Number of samples
    n_dim::Int # Number of features per sample
    n_output::Int # Number of outputs
end

function wrap_modata(X::TX, y::TY) where {TX,TY<:AbstractVector}
    all(length.(y) .== length(X)) || error(
        "There is not the same number of samples in X ($(length(X))) and y ($(length.(y))))",
    )
    Tx = eltype(first(X))
    return MODataContainer{Tx,TX,TY}(X, y, length(X), length(first(X)), length(y))
end

n_output(d::MODataContainer) = d.n_output

function wrap_X(X::AbstractMatrix{T}, obsdim::Int=1) where {T<:Real}
    return KernelFunctions.vec_of_vecs(X; obsdim=obsdim), T
end

function wrap_X(X::AbstractVector{T}, ::Int=1) where {T<:Real}
    return X, T
end

function wrap_X(X::AbstractVector{<:AbstractVector{T}}, ::Int=1) where {T<:Real}
    return X, T
end
