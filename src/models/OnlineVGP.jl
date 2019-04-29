""" Class for sparse variational Gaussian Processes """
mutable struct OnlineVGP{L<:Likelihood,I<:Inference,T<:Real,V<:AbstractVector{T}} <: SparseGP{L,I,T,V}
    # X::Matrix{T} #Feature vectors
    # y::LatentArray #Output (-1,1 for classification, real for regression, matrix for multiclass)
    # nSample::Int64 # Number of data points
    # nDim::Int64 # Number of covariates per data point
    # nFeature::Int64 # Number of features of the GP (equal to number of points)
    nLatent::Int64 # Number of latent GPs
    IndependentPriors::Bool # Use of separate priors for each latent GP
    nPrior::Int64 # Equal to 1 or nLatent given IndependentPriors
    Zalg::ZAlg
    Zupdated::Bool
    Sequential::Bool #Defines if we know how many point will be treated at the beginning
    dataparsed::Bool #Check if all data has been treated
    lastindex::Int64
    μ::LatentArray{V}
    Σ::LatentArray{Symmetric{T,Matrix{T}}}
    η₁::LatentArray{V}
    η₂::LatentArray{Symmetric{T,Matrix{T}}}
    Z::LatentArray{Matrix{T}}
    Kmm::LatentArray{Symmetric{T,Matrix{T}}}
    invKmm::LatentArray{Symmetric{T,Matrix{T}}}
    Knm::LatentArray{Matrix{T}}
    κ::LatentArray{Matrix{T}}
    K̃::LatentArray{V}
    Zₐ::LatentArray{Matrix{T}}
    Kab::LatentArray{Matrix{T}}
    κₐ::LatentArray{Matrix{T}}
    K̃ₐ::LatentArray{V}
    invDₐ::LatentArray{Symmetric{T,Matrix{T}}}
    prevη₁::LatentArray{V}
    prev𝓛ₐ::LatentArray{T}
    kernel::LatentArray{Kernel{T}}
    likelihood::Likelihood{T}
    inference::Inference{T}
    verbose::Int64
    Autotuning::Bool
    atfrequency::Int64
    OptimizeInducingPoints::Bool
    Trained::Bool
end

"""Create a sparse variational Gaussian Process model
Argument list :

**Mandatory arguments**
 - `X` : input features, should be a matrix N×D where N is the number of observation and D the number of dimension
 - `y` : input labels, can be either a vector of labels for multiclass and single output or a matrix for multi-outputs (note that only one likelihood can be applied)
 - `kernel` : covariance function, can be either a single kernel or a collection of kernels for multiclass and multi-outputs models
 - `likelihood` : likelihood of the model, currently implemented : Gaussian, Bernoulli (with logistic link), Multiclass (softmax or logistic-softmax) see [`Likelihood`](@ref)
 - `inference` : inference for the model, can be analytic, numerical or by sampling, check the model documentation to know what is available for your likelihood see [`Inference`](@ref)
 - `nInducingPoints` : number of inducing points
 - `ZAlg` : Algorithm to add automatically inducing points, `CircleKMeans` by default, options are : `OfflineKMeans`, `StreamingKMeans`, `Webscale`
**Optional arguments**
 - `verbose` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
 - `Autotuning` : Flag for optimizing hyperparameters
 - `atfrequency` : Choose how many variational parameters iterations are between hyperparameters optimization
 - `IndependentPriors` : Flag for setting independent or shared parameters among latent GPs
 - `OptimizeInducingPoints` : Flag for optimizing the inducing points locations
 - `ArrayType` : Option for using different type of array for storage (allow for GPU usage)
"""
function OnlineVGP(#X::AbstractArray{T1},y::AbstractArray{T2},
            kernel::Kernel,
            likelihood::LikelihoodType,inference::InferenceType,
            Zalg::ZAlg=CircleKMeans(),Sequential::Bool=false
            ;verbose::Integer=0,Autotuning::Bool=true,atfrequency::Integer=1,
            IndependentPriors::Bool=true, OptimizeInducingPoints::Bool=false,ArrayType::UnionAll=Vector) where {T1<:Real,T2,LikelihoodType<:Likelihood,InferenceType<:Inference}

            # X,y,nLatent,likelihood = check_data!(X,y,likelihood)
            @assert check_implementation(:OnlineVGP,likelihood,inference) "The $likelihood is not compatible or implemented with the $inference"
            nLatent = 1
            nPrior = IndependentPriors ? nLatent : 1
            # nSample = size(X,1); nDim = size(X,2);
            kernel = [deepcopy(kernel) for _ in 1:nPrior];

            dataparsed = false;
            lastindex = 1
            if Sequential
                if typeof(Zalg) <: StreamOnline || typeof(Zalg) <: DataSelection
                    inference.MBIndices = 1:(inference.nSamplesUsed)
                    init!(Zalg,X[inference.MBIndices,:],y[1][inference.MBIndices],kernel[1])
                else
                    inference.MBIndices = 1:(lastindex+inference.nSamplesUsed-1)
                    init!(Zalg,X[inference.MBIndices,:],y[1][inference.MBIndices],kernel[1])
                end
            else
                inference.MBIndices = StatsBase.sample(1:nSample,inference.nSamplesUsed,replace=false) #Sample nSamplesUsed indices for the minibatches
                init!(Zalg,X[inference.MBIndices,:],y[1][inference.MBIndices],kernel[1])
            end
            Zupdated = true;
            nFeature = Zalg.k;
            Z = [copy(Zalg.centers) for _ in 1:nPrior]
            Zₐ = similar.(Z)
            μ = LatentArray([zeros(T1,nFeature) for _ in 1:nLatent]); η₁ = deepcopy(μ);
            Σ = LatentArray([Symmetric(Matrix(Diagonal(one(T1)*I,nFeature))) for _ in 1:nLatent]);
            η₂ = -0.5*inv.(Σ);
            κ = LatentArray([zeros(T1,inference.nSamplesUsed, nFeature) for _ in 1:nPrior])
            Knm = deepcopy(κ)
            K̃ = LatentArray([zeros(T1,inference.nSamplesUsed) for _ in 1:nPrior])
            κₐ = LatentArray([zeros(T1, nFeature, nFeature) for _ in 1:nPrior])
            Kab = deepcopy(κₐ)
            K̃ₐ = LatentArray([zeros(T1, nFeature) for _ in 1:nPrior])
            invDₐ = LatentArray([Symmetric(zeros(T1, nFeature, nFeature)) for _ in 1:nPrior])
            𝓛ₐ  = LatentArray(zeros(nLatent))
            prevη₁  = copy.(η₁)
            Kmm = LatentArray([similar(Σ[1]) for _ in 1:nPrior]); invKmm = similar.(Kmm)
            nSamplesUsed = nSample
            @assert inference.nSamplesUsed > 0 && inference.nSamplesUsed < nSample "The size of mini-batch is incorrect (negative or bigger than number of samples), please set nMinibatch correctly in the inference object"
            nSamplesUsed = inference.nSamplesUsed

            likelihood = init_likelihood(likelihood,inference,nLatent,nSamplesUsed)
            inference = init_inference(inference,nLatent,nFeature,nSample,nSamplesUsed)
            return OnlineVGP{LikelihoodType,InferenceType,T1,ArrayType{T1}}(
                    #X,y,
                    nSample, nDim, nFeature, nLatent,
                    IndependentPriors,nPrior,
                    Zalg,Zupdated,Sequential,dataparsed,lastindex,
                    μ,Σ,η₁,η₂,
                    Z,Kmm,invKmm,Knm,κ,K̃,
                    Zₐ,Kab,κₐ,K̃ₐ,invDₐ,prevη₁,𝓛ₐ,
                    kernel,likelihood,inference,
                    verbose,Autotuning,atfrequency,OptimizeInducingPoints,false)
end

function Base.show(io::IO,model::OnlineVGP{<:Likelihood,<:Inference,T}) where T
    print(io,"Online Variational Gaussian Process with a $(model.likelihood) infered by $(model.inference) ")
end

function updateZ!(model::OnlineVGP)
    model.Zₐ = copy.(model.Z)
    model.invDₐ = Symmetric.(-2.0.*model.η₂.-model.invKmm)
    update!(model.Zalg,model.X[model.inference.MBIndices,:],model.y[1][model.inference.MBIndices],model.kernel[1]) #TEMP FOR 1 latent
    NCenters = model.Zalg.k
    Nnewpoints = NCenters-model.nFeature
    # computeMatrices!(model)
    #Make the latent variables larger #TODO Preallocating them might be a better option
    if Nnewpoints!=0
        # println("Adapting to new number of points")
        # model.μ[1] = vcat(model.μ[1], zeros(Nnewpoints))
        # model.η₁[1] = vcat(model.η₁[1], zeros(Nnewpoints))
        # Σ_temp = Matrix{Float64}(I,NCenters,NCenters)
        # Σ_temp[1:model.nFeature,1:model.nFeature] = model.Σ[1]
        # model.Σ[1] = Symmetric(Σ_temp)
        # η₂temp = Matrix{Float64}(-0.5*I,NCenters,NCenters)
        # η₂temp[1:model.nFeature,1:model.nFeature] = model.η₂[1]
        # model.η₂[1] = Symmetric(η₂temp)
        model.nFeature = NCenters
    end
    model.Zupdated = true
    model.Z = [copy(model.Zalg.centers) for _ in 1:model.nPrior]
end
