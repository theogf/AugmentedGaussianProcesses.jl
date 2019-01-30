""" Class for sparse variational Gaussian Processes """

mutable struct SVGP{L<:Likelihood,I<:Inference,T<:Real,A<:AbstractArray} <: GP{L,I,T,A}
    X::AbstractMatrix #Feature vectors
    y::AbstractVector{AbstractVector} #Output (-1,1 for classification, real for regression, matrix for multiclass)
    nSample::Int64 # Number of data points
    nDim::Int64 # Number of covariates per data point
    nFeature::Int64 # Number of features of the GP (equal to number of points)
    nLatent::Int64 # Number pf latent GPs
    IndependentPriors::Bool # Use of separate priors for each latent GP
    nPrior::Int64 # Equal to 1 or nLatent given IndependentPriors
    Z::AbstractVector{AbstractMatrix} #Inducing points locations
    μ::AbstractVector{AbstractVector}
    Σ::AbstractVector{AbstractMatrix}
    η₁::AbstractVector{AbstractVector}
    η₂::AbstractVector{AbstractMatrix}
    Kmm::AbstractVector{AbstractMatrix}
    invKmm::AbstractVector{AbstractMatrix}
    Knm::AbstractVector{AbstractMatrix}
    κ::AbstractVector{AbstractMatrix}
    K̃::AbstractVector{AbstractVector}
    kernel::AbstractVector{Kernel}
    likelihood::Likelihood
    inference::Inference
    verbose::Int64
    Autotuning::Bool
    atfrequency::Int64
    Trained::Bool
end

function SVGP(X::AbstractArray{T1,N1},y::AbstractArray{T2,N2},kernel::Kernel,
            likelihood::LType,inference::IType,
            nInducingPoints::Integer=0#,Z::Union{AbstractVector{AbstractArray},AbstractArray}=[],
            ;verbose::Integer=0,Autotuning::Bool=true,atfrequency::Integer=1,
            IndependentPriors::Bool=true,
            Stochastic::Bool=false,nMinibatch::Integer=0) where {T1<:Real,T2,N1,N2,LType<:Likelihood,IType<:Inference}

            X,y = check_data!(X,y,likelihood)
            @assert check_implementation(likelihood,inference) "The $likelihood is not compatible or implemented with the $inference"

            nLatent = length(y);
            nPrior = IndependentPriors ? nLatent : 1
            nSample = size(X,1); nDim = size(X,2);
            kernel = [deepcopy(kernel) for _ in 1:nPrior]

            @assert nInducingPoints > 0 && nInducingPoints < nSample "The number of inducing points is incorrect (negative or bigger than number of samples)"
            Z = KMeansInducingPoints(X,nInducingPoints,nMarkov=10); Z=[copy(Z) for _ in 1:nPrior]
            nFeature = nInducingPoints

            μ = [zeros(T1,nFeature) for _ in 1:nLatent]; η₁ = copy(μ)
            Σ = [Symmetric(Array(Diagonal(one(T1)*I,nFeature))) for _ in 1:nLatent];
            η₂ = inv.(Σ)*(-0.5);
            κ = [zeros(T1,Stochastic ? nMinibatch : nSample, nFeature) for _ in 1:nPrior]
            Knm = copy(κ)
            K̃ = [zeros(T1,Stochastic ? nMinibatch : nSample) for _ in 1:nPrior]
            Kmm = [copy(Σ[1]) for _ in 1:nPrior]; invKmm = copy(Kmm)
            #TODO This should be done externally in the initialization of the inference struct
            if Stochastic
                @assert nMinibatch > 0 && nMinibatch < nSample "The size of mini-batch is incorrect (negative or bigger than number of samples), please set nMinibatch"
                inference = typeof(inference)(nSample,nMinibatch,η₁,η₂)
            else
                inference = typeof(inference)(nSample)
            end
            SVGP{LType,IType,T1,AbstractArray{T1,N1}}(X,y,
                    nSample, nDim, nFeature, nLatent,
                    IndependentPriors,nPrior,
                    Z,μ,Σ,η₁,η₂,
                    Kmm,invKmm,Knm,κ,K̃,
                    kernel,likelihood,inference,
                    verbose,Autotuning,atfrequency,false)
end
