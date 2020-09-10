abstract type AbstractDataContainer end

struct DataContainer{
    Tx<:Real,
    TX<:AbstractVector,
    Ty<:Real,
    TY<:AbstractVector,
} <: AbstractDataContainer
    X::TX # Feature vectors
    y::TY # Output (-1,1 for classification, real for regression, matrix for multiclass)
    nSamples::Int # Number of samples
    nDim::Int # Number of features per sample
end

function wrap_data(X::TX, y::TY) where {TX, TY<:AbstractVector{<:Real}}
    size(y, 1) == size(X, 1) || error("There is not the same number of samples in X ($(length(TX))) and y ($(size(y, 1)))")
    Tx = eltype(first(X))
    Ty = eltype(first(y))
    return DataContainer{Tx, TX, Ty, TY}(X, y, length(X), length(first(X)))
end

function wrap_data(X::TX, y::TY) where {TX, TY}
    size(first(y), 1) == size(X, 1) || error("There is not the same number of samples in X ($(length(TX))) and y ($(size(y, 1)))")
    Tx = eltype(first(X))
    Ty = eltype(first(y))
    return DataContainer{Tx, TX, Ty, TY}(X, y, length(X), length(first(X)))
end

function wrap_X(X::AbstractMatrix{T}, obsdim = 1) where {T<:Real}
    return KernelFunctions.vec_of_vecs(X, obsdim = obsdim), T
end

function wrap_X(X::AbstractVector{T}, obsdim = 1) where {T<:Real}
    return wrap_X(reshape(X, :, 1), 1)
end

function wrap_X(X::AbstractVector{<:AbstractVector{T}}, obsdim = 1) where {T<:Real}
    return X, T
end

nSamples(d::AbstractDataContainer) = d.nSamples
nDim(d::AbstractDataContainer) = d.nDim

input(d::AbstractDataContainer) = d.X
output(d::AbstractDataContainer) = d.y

mutable struct OnlineDataContainer <: AbstractDataContainer
    X::AbstractVector # Feature vectors
    y::AbstractVector # Output (-1,1 for classification, real for regression, matrix for multiclass)
    nSamples::Int # Number of samples
    nDim::Int # Number of features per sample
    function OnlineDataContainer()
        return new()
    end
end



function wrap_data!(data::OnlineDataContainer, X::AbstractVector, y::AbstractVector)
    data.X = X
    data.y = y
    data.nSamples = size(X, 1)
    data.nDim = size(first(X), 1)
end