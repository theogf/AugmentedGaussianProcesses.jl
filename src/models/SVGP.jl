"""
Class for sparse variational Gaussian Processes

```julia
SVGP(X::AbstractArray{T1},y::AbstractVector{T2},kernel::Kernel,
    likelihood::LikelihoodType,inference::InferenceType, nInducingPoints::Int;
    verbose::Int=0,optimizer::Union{Optimizer,Nothing,Bool}=Adam(α=0.01),atfrequency::Int=1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
    Zoptimizer::Union{Optimizer,Nothing,Bool}=false,
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
mutable struct SVGP{T<:Real,TLikelihood<:Likelihood{T},TInference<:Inference,N} <: AbstractGP{T,TLikelihood,TInference,N}
    X::Matrix{T} #Feature vectors
    y::Vector #Output (-1,1 for classification, real for regression, matrix for multiclass)
    nSamples::Int64 # Number of data points
    nDim::Int64 # Number of covariates per data point
    nFeatures::Int64 # Number of features of the GP (equal to number of points)
    nLatent::Int64 # Number pf latent GPs
    f::NTuple{N,_SVGP}
    likelihood::TLikelihood
    inference::TInference
    verbose::Int64
    atfrequency::Int64
    Trained::Bool
end

function SVGP(X::AbstractArray{T1},y::AbstractVector{T2},kernel::Kernel,
            likelihood::TLikelihood,inference::TInference, nInducingPoints::Int;
            verbose::Int=0,optimizer=Flux.ADAM(0.01),atfrequency::Int=1,
            mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(), variance::Real = 1.0,
            Zoptimizer::Union{Optimizer,Nothing,Bool}=false,
            ArrayType::UnionAll=Vector) where {T1<:Real,T2,TLikelihood<:Likelihood,TInference<:Inference}

            X,y,nLatent,likelihood = check_data!(X,y,likelihood)
            @assert check_implementation(:SVGP,likelihood,inference) "The $likelihood is not compatible or implemented with the $inference"

            nSamples = size(X,1); nDim = size(X,2);
            if isa(optimizer,Bool)
                optimizer = optimizer ? Flux.ADAM(0.01) : nothing
            end

            @assert nInducingPoints > 0 "The number of inducing points is incorrect (negative or bigger than number of samples)"
            if nInducingPoints > nSamples
                @warn "Number of inducing points bigger than the number of points : reducing it to the number of samples: $(nSamples)"
                nInducingPoints = nSamples
            end
            if nInducingPoints == nSamples
                Z = X
            else
                Z = KMeansInducingPoints(X,nInducingPoints,nMarkov=10)
            end
            if isa(Zoptimizer,Bool)
                Zoptimizer = Zoptimizer ? ADAM(α=0.001) : nothing
            end
            Z = InducingPoints(Z,Zoptimizer)
            nFeatures = nInducingPoints

            if typeof(mean) <: Real
                mean = ConstantMean(mean)
            elseif typeof(mean) <: AbstractVector{<:Real}
                mean = EmpiricalMean(mean)
            end

            nMinibatch = nSamples
            if inference.Stochastic
                @assert inference.nMinibatch > 0 && inference.nMinibatch < nSamples "The size of mini-batch $(inference.nMinibatch) is incorrect (negative or bigger than number of samples), please set nMinibatch correctly in the inference object"
                nMinibatch = inference.nMinibatch
            end

            latentf = ntuple( _ -> _SVGP{T1}(nFeatures,nMinibatch,Z,kernel,mean,variance,optimizer),nLatent)

            likelihood = init_likelihood(likelihood,inference,nLatent,nMinibatch,nFeatures)
            inference = tuple_inference(inference,nLatent,nFeatures,nSamples,nMinibatch)
            inference.xview = view(X,1:nMinibatch,:)
            inference.yview = view_y(likelihood,y,1:nMinibatch)

            model = SVGP{T1,TLikelihood,typeof(inference),_SVGP{T1},nLatent}(X,y,
                    nSamples, nDim, nFeatures, nLatent,
                    latentf,likelihood,inference,
                    verbose,atfrequency,false)
            # if isa(inference.optimizer,ALRSVI)
                # init!(model.inference,model)
            # end
            # return model
end

function Base.show(io::IO,model::SVGP{T,<:Likelihood,<:Inference}) where {T}
    print(io,"Sparse Variational Gaussian Process with a $(model.likelihood) infered by $(model.inference) ")
end

get_y(model::SVGP) = model.inference.yview
get_Z(model::SVGP) = getproperty.(getproperty.(model.f,:Z),:Z)
