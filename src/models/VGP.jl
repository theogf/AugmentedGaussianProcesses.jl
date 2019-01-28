""" Class for variational Gaussian Processes models"""

mutable struct VGP{L<:Likelihood,I<:Inference,T<:Real,A<:AbstractArray} <: GP{L,I,T,A}
    X::AbstractVector{AbstractMatrix} #Feature vectors
    y::AbstractVector{AbstractVector} #Output (-1,1 for classification, real for regression, matrix for multiclass)
    nSample::Int64 # Number of data points
    nDim::Int64 # Number of covariates per data point
    nFeature::Int64 # Number of features of the GP (equal to number of points)
    nLatent::Int64 # Number pf latent GPs
    IndependentPriors::Bool # Use of separate priors for each latent GP
    nPrior::Int64 # Equal to 1 or nLatent given IndependentPriors
    μ::AbstractVector{AbstractVector}
    Σ::AbstractVector{AbstractMatrix}
    η₁::AbstractVector{AbstractVector}
    η₂::AbstractVector{AbstractMatrix}
    Knn::AbstractVector{AbstractMatrix}
    invKnn::AbstractVector{AbstractMatrix}
    kernel::AbstractVector{Kernel}
    likelihood::Likelihood
    inference::Inference
    verbose::Int64 #Level of printing information
end

function VGP(X::AbtractArray{T1,N1},y::AbstractArray{T2,N2},kernel::Kernel,
            likelihood::L,inference::I;
            verbose::Integer=0,Autotuning::Bool=true,
            IndependentPriors::Bool=true) where {T1<:Real,T2,N1,N2,L<:Likelihood,I<:Inference}
            checkdata!(X,y,likelihood)
            checkimplementation(likelihood,inference)
            nLatent = length(y);
            nPrior = IndependentPriors ? nLatent : 1
            nFeature = nSample = size(X,1); nDim = size(X,2);
            μ = [zeros(T1,nFeature) for _ in 1:nLatent]; η₁ = copy(μ)
            Σ = [Symmetric(Diagonal(one(T1)*I,nFeature)) for _ in 1:nLatent];
            η₂ = inv.(Σ)*(-0.5);
            Knn = copy(Σ); invKnn = copy(Σ)
            VGP{L,I,T1,AbstractArray{T1,N1}}(X,y,nFeature, nDim, nFeaturem nLatent,
                    IndependentPriors,nPrior,μ,Σ,η₁,η₂,
                    Knn,invKnn,kernel,likelihood,inference)
end
