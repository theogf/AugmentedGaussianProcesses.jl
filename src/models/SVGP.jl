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
    OptimizeInducingPoints::Bool
    Trained::Bool
end

function SVGP(X::AbstractArray{T1,N1},y::AbstractArray{T2,N2},kernel::Kernel,
            likelihood::LType,inference::IType,
            nInducingPoints::Integer=0#,Z::Union{AbstractVector{AbstractArray},AbstractArray}=[],
            ;verbose::Integer=0,Autotuning::Bool=true,atfrequency::Integer=1,
            IndependentPriors::Bool=true, OptimizeInducingPoints::Bool=false) where {T1<:Real,T2,N1,N2,LType<:Likelihood,IType<:Inference}

            X,y,likelihood = check_data!(X,y,likelihood)
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
            κ = [zeros(T1,inference.Stochastic ? inference.nSamplesUsed : nSample, nFeature) for _ in 1:nPrior]
            Knm = copy(κ)
            K̃ = [zeros(T1,inference.Stochastic ? inference.nSamplesUsed : nSample) for _ in 1:nPrior]
            Kmm = [copy(Σ[1]) for _ in 1:nPrior]; invKmm = copy(Kmm)
            nSamplesUsed = nSample
            if inference.Stochastic
                @assert inference.nSamplesUsed > 0 && inference.nSamplesUsed < nSample "The size of mini-batch is incorrect (negative or bigger than number of samples), please set nMinibatch correctly in the inference object"
                nSamplesUsed = inference.nSamplesUsed
            end

            likelihood = init_likelihood(likelihood,nLatent,nSamplesUsed)
            inference = init_inference(inference,nLatent,nFeature,nSample,nSamplesUsed)
            model = SVGP{LType,IType,T1,AbstractArray{T1,N1}}(X,y,
                    nSample, nDim, nFeature, nLatent,
                    IndependentPriors,nPrior,
                    Z,μ,Σ,η₁,η₂,
                    Kmm,invKmm,Knm,κ,K̃,
                    kernel,likelihood,inference,
                    verbose,Autotuning,atfrequency,OptimizeInducingPoints,false)
            if isa(inference.optimizer_η₁[1],ALRSVI)
                init!(model.inference,model)
                return model
            else
                return model
            end
end

function Base.show(io::IO,model::SVGP{<:Likelihood,<:Inference,T}) where T
    print(io,"Sparse Variational Gaussian Process with a $(model.likelihood) infered by $(model.inference) ")
end
