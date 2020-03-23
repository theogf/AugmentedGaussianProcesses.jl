"""
Class for multi-output sparse variational Gaussian Processes based on the Linear Coregionalization Model (LCM)

```julia
MOSVGP(X::AbstractArray{T},y::AbstractVector{AbstractArray{T}},kernel::Kernel,
    likelihood::AbstractVector{Likelihoods},inference::InferenceType, nInducingPoints::Int;
    verbose::Int=0,optimiser=ADAM(0.001),atfrequency::Int=1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(),
    Zoptimiser=false,
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
- `optimiser` : Optimiser used for the kernel parameters. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `ADAM(0.001)`
 - `atfrequency` : Choose how many variational parameters iterations are between hyperparameters optimization
 - `mean` : PriorMean object, check the documentation on it [`MeanPrior`](@ref meanprior)
 - `IndependentPriors` : Flag for setting independent or shared parameters among latent GPs
- `Zoptimiser` : Optimiser used for inducing points locations. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `ADAM(0.001)`
 - `ArrayType` : Option for using different type of array for storage (allow for GPU usage)
"""
mutable struct MOSVGP{T<:Real,TLikelihood<:Likelihood{T},TInference<:Inference,N,Q} <: AbstractGP{T,TLikelihood,TInference,N}
    X::Vector{Matrix{T}} #Feature vectors
    y::Vector #Output (-1,1 for classification, real for regression, matrix for multiclass)
    nSamples::Int64 # Number of data points
    nDim::Int64 # Number of covariates per data point
    nFeatures::Int64 # Number of features of the GP (equal to number of points)
    nLatent::Int64 # Number of latent GPs
    nX::Int64
    nTask::Int64
    nf_per_task::Vector{Int64}
    f::NTuple{Q,_SVGP}
    likelihood::Vector{TLikelihood}
    inference::TInference
    A::Array{T,3}
    A_opt
    verbose::Int64
    atfrequency::Int64
    Trained::Bool
end



function MOSVGP(
            X::AbstractArray{T},y::AbstractVector{<:AbstractVector},kernel::Kernel,
            likelihood::TLikelihood,inference::TInference,nLatent::Int,nInducingPoints::Int;
            verbose::Int=0,optimiser=ADAM(0.01),atfrequency::Int=1,
            mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(), variance::Real = 1.0,Aoptimiser=ADAM(0.01),
            Zoptimiser=false,
            ArrayType::UnionAll=Vector) where {T<:Real,TLikelihood<:Likelihood,TInference<:Inference}

            @assert length(y) > 0 "y should not be an empty vector"
            nTask = length(y)
            likelihoods = [deepcopy(likelihood) for _ in 1:nTask]

            X = wrap_X_multi(X, nTask)

            nf_per_task = zeros(Int64,nTask)
            corrected_y = Vector(undef,nTask)
            for i in 1:nTask
                corrected_y[i],nf_per_task[i],likelihoods[i] = check_data!(X,y[i],likelihoods[i])
            end

            @assert inference isa AnalyticVI "The inference object should be of type `AnalyticVI`"
            @assert implemented(likelihood, inference) "The $likelihood is not compatible or implemented with the $inference"

            nSamples = size(X,1); nDim = size(X,2);
            if isa(optimiser,Bool)
                optimiser = optimiser ? ADAM(0.01) : nothing
            end
            if isa(AOptimiser,Bool)
                Aoptimiser = Aoptimiser ? ADAM(0.01) : nothing
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
                Zoptimizer = Zoptimizer ? Adam(α=0.01) : nothing
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

            latent_f = ntuple( _ -> _SVGP{T}(nFeatures,nMinibatch,Z,kernel,mean,variance,optimiser),nLatent)

            A = [[randn(nLatent) |> x->x/sqrt(sum(abs2,x)) for i in 1:nf_per_task[j]] for j in 1:nTask]

            likelihoods .= init_likelihood.(likelihoods,inference,nf_per_task,nMinibatch,nFeatures)
            inference = tuple_inference(inference,nLatent,nFeatures,nSamples,nMinibatch)
            inference.xview = view.(X, range.(1, nMinibatch, step = 1), :)
            inference.yview = view(y, :)

            model = MOSVGP{T,TLikelihood,typeof(inference),nTask,nLatent}(X,corrected_y,
                    nSamples, nDim, nFeatures, nLatent,
                    nTask, nf_per_task,
                    latent_f,likelihoods,inference,A,Aoptimizer,
                    verbose,atfrequency,false)
            # if isa(inference.optimizer,ALRSVI)
                # init!(model.inference,model)
            # end
            # return model
end

function Base.show(io::IO,model::MOSVGP{T,<:Likelihood,<:Inference}) where {T}
    print(io,"Multioutput Sparse Variational Gaussian Process with the likelihoods $(model.likelihood) infered by $(model.inference) ")
end

@traitimpl IsMultiOutput{MOSVGP}

get_X(model::MOSVGP) = model.X
get_Z(model::MOSVGP) = get_Z.(model.f)
objective(model::MOSVGP) = ELBO(model)
