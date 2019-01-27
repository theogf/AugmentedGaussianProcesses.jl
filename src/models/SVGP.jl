""" Class for sparse variational Gaussian Processes """

mutable struct SVGP{L<:Likelihood,I<:Inference,T<:Real,A<:AbstractArray} <: GP{L,I,T,A}
    X::AbstractMatrix{T} #Feature vectors
    y::Vector #Output (-1,1 for classification, real for regression, matrix for multiclass)
    nSample::Int64 # Number of data points
    nDim::Int64 # Number of covariates per data point
    nFeature::Int64 # Number of features of the GP (equal to number of points)
    nLatent::Int64 # Number pf latent GPs
    IndependentPriors::Bool # Use of separate priors for each latent GP
    nPrior::Int64 # Equal to 1 or nLatent given IndependentPriors
    Z::AbstractVector{AbstractMatrix} #Inducing points locations
    μ::AbstractVector{AbstractVector}
    Σ::AbstractVector{AbstractMatrix}
    Kmm::AbstractVector{AbstractMatrix}
    invKmm::AbstractVector{AbstractMatrix}
    Knm::AbstractVector{AbstractMatrix}
    κ::AbstractVector{AbstractMatrix}
    K̃::AbstractVector{AbstractVector}
    likelihood::Likelihood
    inference::Inference
end
