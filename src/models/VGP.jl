""" Class for variational Gaussian Processes models"""

mutable struct VGP{L<:Likelihood,I<:Inference,T<:Real,A<:AbstractArray} <: GP{L,I,T,A}
    X::AbstractMatrix{T} #Feature vectors
    y::Vector #Output (-1,1 for classification, real for regression, matrix for multiclass)
    nSample::Int64 # Number of data points
    nDim::Int64 # Number of covariates per data point
    nFeature::Int64 # Number of features of the GP (equal to number of points)
    nLatent::Int64 # Number pf latent GPs
    μ::AbstractVector{AbstractVector}
    Σ::AbstractVector{AbstractMatrix}
    Knn::AbstractVector{AbstractMatrix}
    invKnn::AbstractVector{AbstractMatrix}
    kernel::AbstractVector{Kernel}
    likelihood::Likelihood
    inference::Inference
    verbose::Int64 #Level of printing information
    function VGP()

    end
end

function VGP(X::AbtractArray)
