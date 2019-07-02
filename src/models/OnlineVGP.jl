""" Class for sparse variational Gaussian Processes """
mutable struct OnlineVGP{L<:Likelihood,I<:Inference,T<:Real,V<:AbstractVector{T}} <: SparseGP{L,I,T,V}
    X::Matrix{T} #Feature vectors
    y::LatentArray #Output (-1,1 for classification, real for regression, matrix for multiclass)
    nDim::Int64 # Number of covariates per data point
    nFeatures::Int64 # Number of features of the GP (equal to number of points)
    nLatent::Int64 # Number of latent GPs
    IndependentPriors::Bool # Use of separate priors for each latent GP
    nPrior::Int64 # Equal to 1 or nLatent given IndependentPriors
    kernel::LatentArray{Kernel{T}}
    likelihood::Likelihood{T}
    inference::Inference{T}
    Zalg::ZAlg
    Zupdated::Bool
    μ::LatentArray{V}
    Σ::LatentArray{Symmetric{T,Matrix{T}}}
    η₁::LatentArray{V}
    η₂::LatentArray{Symmetric{T,Matrix{T}}}
    μ₀::LatentArray{PriorMean{T}}
    Z::LatentArray{Matrix{T}}
    Kmm::LatentArray{Symmetric{T,Matrix{T}}}
    invKmm::LatentArray{Symmetric{T,Matrix{T}}}
    Knm::LatentArray{Matrix{T}}
    κ::LatentArray{Matrix{T}}
    κold::LatentArray{Matrix{T}}
    K̃::LatentArray{V}
    Zₐ::LatentArray{Matrix{T}}
    Kab::LatentArray{Matrix{T}}
    κₐ::LatentArray{Matrix{T}}
    K̃ₐ::LatentArray{Matrix{T}}
    invDₐ::LatentArray{Symmetric{T,Matrix{T}}}
    prevη₁::LatentArray{V}
    prev𝓛ₐ::LatentArray{T}
    verbose::Int64
    Autotuning::Bool
    atfrequency::Int64
    Zoptimizer::Union{LatentArray{Optimizer},Nothing}
    Trained::Bool
end

"""Create a sparse variational Gaussian Process model
Argument list :

**Mandatory arguments**
 # - `X` : input features, should be a matrix N×D where N is the number of observation and D the number of dimension
 # - `y` : input labels, can be either a vector of labels for multiclass and single output or a matrix for multi-outputs (note that only one likelihood can be applied)
 - `kernel` : covariance function, can be either a single kernel or a collection of kernels for multiclass and multi-outputs models
 - `likelihood` : likelihood of the model, currently implemented : Gaussian, Bernoulli (with logistic link), Multiclass (softmax or logistic-softmax) see [`Likelihood`](@ref)
 - `inference` : inference for the model, can be analytic, numerical or by sampling, check the model documentation to know what is available for your likelihood see [`Inference`](@ref)
 # - `nInducingPoints` : number of inducing points
 - `ZAlg` : Algorithm to add automatically inducing points, `CircleKMeans` by default, options are : `OfflineKMeans`, `StreamingKMeans`, `Webscale`
**Optional arguments**
 - `verbose` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
 - `Autotuning` : Flag for optimizing hyperparameters
 - `atfrequency` : Choose how many variational parameters iterations are between hyperparameters optimization
 - `IndependentPriors` : Flag for setting independent or shared parameters among latent GPs
 - `Zoptimizer` : Optimizer for the inducing points locations
 - `ArrayType` : Option for using different type of array for storage (allow for GPU usage)
"""
function OnlineVGP(#X::AbstractArray{T1},y::AbstractArray{T2},
            kernel::Kernel,
            likelihood::Likelihood{T1},inference::Inference,
            Zalg::ZAlg=CircleKMeans()#,Sequential::Bool=false
            ;verbose::Integer=0,Autotuning::Bool=true,atfrequency::Integer=1,
            mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
            IndependentPriors::Bool=true, Zoptimizer::Union{Any,Nothing}=Nothing(),ArrayType::UnionAll=Vector) where {T1<:Real,T2}

            @assert check_implementation(:OnlineVGP,likelihood,inference) "The $likelihood is not compatible or implemented with the $inference"
            nLatent = 1
            nPrior = IndependentPriors ? nLatent : 1
            kernel = [deepcopy(kernel) for _ in 1:nPrior];
            Zupdated = false;
            μ = LatentArray{ArrayType{T1}}()
            Σ = LatentArray{Symmetric{T1,Matrix{T1}}}()
            η₁ = LatentArray{ArrayType{T1}}()
            η₂ = LatentArray{Symmetric{T1,Matrix{T1}}}()
            μ₀ = []
            if typeof(mean) <: Real
                μ₀ = [ConstantMean(mean)]
            elseif typeof(mean) <: AbstractVector{<:Real}
                μ₀ = [EmpiricalMean(mean)]
            else
                μ₀ = [mean]
            end
            Z = LatentArray{Matrix{T1}}()
            if !isnothing(Zoptimizer)
                Zoptimizer = [deepcopy(Zoptimizer) for _ in 1:nPrior]
            end
            Kmm = LatentArray{Symmetric{T1,Matrix{T1}}}()
            invKmm = LatentArray{Symmetric{T1,Matrix{T1}}}()
            Knm = LatentArray{Matrix{T1}}()
            κ = LatentArray{Matrix{T1}}()
            κold = LatentArray{Matrix{T1}}()
            K̃ = LatentArray{ArrayType{T1}}()
            Zₐ = LatentArray{Matrix{T1}}()
            Kab = LatentArray{Matrix{T1}}()
            κₐ = LatentArray{Matrix{T1}}()
            K̃ₐ = LatentArray{Matrix{T1}}()
            invDₐ = LatentArray{Symmetric{T1,Matrix{T1}}}()
            prevη₁ = LatentArray{ArrayType{T1}}()
            𝓛ₐ = LatentArray{T1}()
            inference.nIter = 1
            return OnlineVGP{typeof(likelihood),typeof(inference),T1,ArrayType{T1}}(
                    Matrix{T1}(undef,1,1),LatentArray(),
                    # nSample, nDim, nFeatures,
                    -1,0,nLatent,
                    IndependentPriors,nPrior,
                    kernel,likelihood,inference,
                    Zalg,Zupdated,
                    # Sequential,dataparsed,lastindex,
                    μ,Σ,η₁,η₂,μ₀,
                    Z,Kmm,invKmm,Knm,κ,κold,K̃,
                    Zₐ,Kab,κₐ,K̃ₐ,invDₐ,prevη₁,𝓛ₐ,
                    verbose,Autotuning,atfrequency,Zoptimizer,false
                    )
            # model.verbose = verbose;
            # model.Autotuning = Autotuning;
            # model.atfrequency = atfrequency;
            # model.OptimizeInducingPoints = OptimizeInducingPoints
            # model.Trained=false
end

function Base.show(io::IO,model::OnlineVGP{<:Likelihood,<:Inference,T}) where T
    print(io,"Online Variational Gaussian Process with a $(model.likelihood) infered by $(model.inference) ")
end
