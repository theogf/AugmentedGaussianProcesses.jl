"""
Class for sparse variational Gaussian Processes

```julia
SVGP(X::AbstractArray{T₁},y::AbstractArray{T₂},kernel::Union{Kernel,AbstractVector{<:Kernel}},
    likelihood::LikelihoodType,inference::InferenceType, nInducingPoints::Int;
    verbose::Int=0,optimizer::Union{Optimizer,Nothing,Bool}=Adam(α=0.01),atfrequency::Int=1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
    IndependentPriors::Bool=true,Zoptimizer::Union{Optimizer,Nothing,Bool}=false,
    ArrayType::UnionAll=Vector)
```

Argument list :

**Mandatory arguments**
 - `X` : input features, should be a matrix N×D where N is the number of observation and D the number of dimension
 - `y` : input labels, can be either a vector of labels for multiclass and single output or a matrix for multi-outputs (note that only one likelihood can be applied)
 - `kernel` : covariance function, can be either a single kernel or a collection of kernels for multiclass and multi-outputs models
 - `likelihood` : likelihood of the model, currently implemented : Gaussian, Student-T, Laplace, Bernoulli (with logistic link), Bayesian SVM, Multiclass (softmax or logistic-softmax) see [`Likelihood`](@ref likelihood_user)
 - `inference` : inference for the model, can be analytic, numerical or by sampling, check the model documentation to know what is available for your likelihood see the [`Compatibility table`](@ref compat_table)
 - `nInducingPoints` : number of inducing points
**Optional arguments**
 - `verbose` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
 - `optimizer` : Optimizer for kernel hyperparameters (to be selected from [GradDescent.jl](https://github.com/jacobcvt12/GradDescent.jl)) or set it to `false` to keep hyperparameters fixed
 - `atfrequency` : Choose how many variational parameters iterations are between hyperparameters optimization
 - `mean` : PriorMean object, check the documentation on it [`MeanPrior`](@ref meanprior)
 - `IndependentPriors` : Flag for setting independent or shared parameters among latent GPs
 - `optimizer` : Optimizer for inducing point locations (to be selected from [GradDescent.jl](https://github.com/jacobcvt12/GradDescent.jl))
 - `ArrayType` : Option for using different type of array for storage (allow for GPU usage)
"""
mutable struct SVGP{T<:Real,TLikelihood<:Likelihood{T},TInference<:Inference,V<:AbstractVector{T}} <: AbstractGP{T,TLikelihood,TInference,V}
    X::Matrix{T} #Feature vectors
    y::LatentArray #Output (-1,1 for classification, real for regression, matrix for multiclass)
    nSample::Int64 # Number of data points
    nDim::Int64 # Number of covariates per data point
    nFeatures::Int64 # Number of features of the GP (equal to number of points)
    nLatent::Int64 # Number pf latent GPs
    IndependentPriors::Bool # Use of separate priors for each latent GP
    nPrior::Int64 # Equal to 1 or nLatent given IndependentPriors
    Z::LatentArray{Matrix{T}} #Inducing points locations
    μ::LatentArray{V}
    Σ::LatentArray{Symmetric{T,Matrix{T}}}
    η₁::LatentArray{V}
    η₂::LatentArray{Symmetric{T,Matrix{T}}}
    μ₀::LatentArray{PriorMean{T}}
    Kmm::LatentArray{Symmetric{T,Matrix{T}}}
    invKmm::LatentArray{Symmetric{T,Matrix{T}}}
    Knm::LatentArray{Matrix{T}}
    κ::LatentArray{Matrix{T}}
    K̃::LatentArray{V}
    kernel::LatentArray{Kernel{T}}
    likelihood::TLikelihood
    inference::TInference
    verbose::Int64
    optimizer::Union{Optimizer,Nothing}
    atfrequency::Int64
    Zoptimizer::Union{Optimizer,Nothing}
    Trained::Bool
end

function SVGP(X::AbstractArray{T₁},y::AbstractArray{T₂},kernel::Union{Kernel,AbstractVector{<:Kernel}},
            likelihood::TLikelihood,inference::TInference, nInducingPoints::Int;
            verbose::Int=0,optimizer::Union{Optimizer,Nothing,Bool}=Adam(α=0.01),atfrequency::Int=1,
            mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
            IndependentPriors::Bool=true,Zoptimizer::Union{Optimizer,Nothing,Bool}=false,
            ArrayType::UnionAll=Vector) where {T₁<:Real,T₂,TLikelihood<:Likelihood,TInference<:Inference}

            X,y,nLatent,likelihood = check_data!(X,y,likelihood)
            @assert check_implementation(:SVGP,likelihood,inference) "The $likelihood is not compatible or implemented with the $inference"

            nPrior = IndependentPriors ? nLatent : 1
            nSample = size(X,1); nDim = size(X,2);
            if isa(optimizer,Bool)
                optimizer = optimizer ? Adam(α=0.01) : nothing
            end
            if !isnothing(optimizer)
                setoptimizer!(kernel,optimizer)
            end
            kernel = [deepcopy(kernel) for _ in 1:nPrior]


            @assert nInducingPoints > 0 && nInducingPoints < nSample "The number of inducing points is incorrect (negative or bigger than number of samples)"
            Z = KMeansInducingPoints(X,nInducingPoints,nMarkov=10); Z=[deepcopy(Z) for _ in 1:nPrior]
            Zoptimizer = !Zoptimizer ? nothing : Zoptimizer
            nFeatures = nInducingPoints


            μ = LatentArray([zeros(T₁,nFeatures) for _ in 1:nLatent]); η₁ = deepcopy(μ);
            Σ = LatentArray([Symmetric(Matrix(Diagonal(one(T₁)*I,nFeatures))) for _ in 1:nLatent]);
            η₂ = -0.5*inv.(Σ);
            κ = LatentArray([zeros(T₁,inference.Stochastic ? inference.nSamplesUsed : nSample, nFeatures) for _ in 1:nPrior])
            Knm = deepcopy(κ)
            K̃ = LatentArray([zeros(T₁,inference.Stochastic ? inference.nSamplesUsed : nSample) for _ in 1:nPrior])
            Kmm = LatentArray([similar(Σ[1]) for _ in 1:nPrior]); invKmm = similar.(Kmm)
            μ₀ = []
            if typeof(mean) <: Real
                μ₀ = [ConstantMean(mean) for _ in 1:nPrior]
            elseif typeof(mean) <: AbstractVector{<:Real}
                μ₀ = [EmpiricalMean(mean) for _ in 1:nPrior]
            else
                μ₀ = [mean for _ in 1:nPrior]
            end

            nSamplesUsed = nSample
            if inference.Stochastic
                @assert inference.nSamplesUsed > 0 && inference.nSamplesUsed < nSample "The size of mini-batch is incorrect (negative or bigger than number of samples), please set nMinibatch correctly in the inference object"
                nSamplesUsed = inference.nSamplesUsed
                opt = kernel[1].fields.variance.opt
                if isa(opt,Adam)
                    opt.α = opt.α*0.1
                end
                setoptimizer!.(kernel,[copy(opt) for _ in 1:nLatent])
            end

            likelihood = init_likelihood(likelihood,inference,nLatent,nSamplesUsed,nFeatures)
            inference = init_inference(inference,nLatent,nFeatures,nSample,nSamplesUsed)
            inference.x = view(X,1:nSample,:)
            inference.y = view.(y,:)

            model = SVGP{T₁,TLikelihood,TInference,ArrayType{T₁}}(X,y,
                    nSample, nDim, nFeatures, nLatent,
                    IndependentPriors,nPrior,
                    Z,μ,Σ,η₁,η₂,
                    μ₀,Kmm,invKmm,Knm,κ,K̃,
                    kernel,likelihood,inference,
                    verbose,optimizer,atfrequency,Zoptimizer,false)
            if isa(inference.optimizer[1],ALRSVI)
                init!(model.inference,model)
            end
            return model
end

function Base.show(io::IO,model::SVGP{T,<:Likelihood,<:Inference}) where {T}
    print(io,"Sparse Variational Gaussian Process with a $(model.likelihood) infered by $(model.inference) ")
end

@inline invK(model::SVGP) = model.invKmm
@inline invK(model::SVGP,i::Integer) = model.invKmm[i]
