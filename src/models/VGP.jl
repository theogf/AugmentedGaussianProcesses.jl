""" Class for variational Gaussian Processes models"""

mutable struct VGP{L<:Likelihood,I<:Inference,T<:Real,A<:AbstractArray} <: GP{L,I,T,A}
    X::AbstractMatrix #Feature vectors
    y::AbstractVector{AbstractVector} #Output (-1,1 for classification, real for regression, matrix for multiclass)
    nSample::Int64 # Number of data points
    nDim::Int64 # Number of covariates per data point
    nFeature::Int64 # Number of features of the GP (equal to number of points)
    nLatent::Int64 # Number pf latent GPs
    IndependentPriors::Bool # Use of separate priors for each latent GP
    nPrior::Int64 # Equal to 1 or nLatent given IndependentPriors
    μ::AbstractVector{AbstractVector{T}}
    Σ::AbstractVector{AbstractMatrix{T}}
    η₁::AbstractVector{AbstractVector{T}}
    η₂::AbstractVector{AbstractMatrix{T}}
    Knn::AbstractVector{AbstractMatrix{T}}
    invKnn::AbstractVector{AbstractMatrix{T}}
    kernel::AbstractVector{Kernel}
    likelihood::Likelihood{T}
    inference::Inference{T}
    verbose::Int64 #Level of printing information
    Autotuning::Bool
    atfrequency::Int64
    Trained::Bool
end

function VGP(X::AbstractArray{T1,N1},y::AbstractArray{T2,N2},kernel::Kernel,
            likelihood::LType,inference::IType;
            verbose::Integer=0,Autotuning::Bool=true,atfrequency::Integer=1,
            IndependentPriors::Bool=true) where {T1<:Real,T2,N1,N2,LType<:Likelihood,IType<:Inference}

            X,y,likelihood = check_data!(X,y,likelihood)
            @assert check_implementation(likelihood,inference) "The $likelihood is not compatible or implemented with the $inference"

            nLatent = length(y);
            nPrior = IndependentPriors ? nLatent : 1
            nFeature = nSample = size(X,1); nDim = size(X,2);
            kernel = [deepcopy(kernel) for _ in 1:nPrior]

            μ = [zeros(T1,nFeature) for _ in 1:nLatent]; η₁ = copy.(μ)
            Σ = [Symmetric(diagm(0=>ones(T1,nFeature))) for _ in 1:nLatent]
            η₂ = -0.5*inv.(Σ);
            Knn = [copy(Σ[1]) for _ in 1:nPrior]; invKnn = copy.(Knn)

            likelihood = init_likelihood(likelihood,nLatent,nSample)
            inference = init_inference(inference,nLatent,nSample,nSample,nSample)

            VGP{LType,IType,T1,AbstractArray{T1,N1}}(X,y,
                    nFeature, nDim, nFeature, nLatent,
                    IndependentPriors,nPrior,μ,Σ,η₁,η₂,
                    Knn,invKnn,kernel,likelihood,inference,
                    verbose,Autotuning,atfrequency,false)
end


"""Basic displaying function"""
function Base.show(io::IO,model::VGP{<:Likelihood,<:Inference,T}) where T
    print(io,"Variational Gaussian Process with a $(model.likelihood) infered by $(model.inference) ")
end
