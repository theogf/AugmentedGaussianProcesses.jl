"""
Class for multi-output sparse variational Gaussian Processes

```julia
MOARGP(X::Vector{AbstractArray{T}},y::AbstractVector{AbstractArray{T}},kernel::Kernel,
    likelihood::AbstractVector{Likelihoods},inference::InferenceType, nInducingPoints::Int;
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
 - `optimizer` : Optimizer for kernel hyperparameters (to be selected from [GradDescent.jl](https://github.com/jacobcvt12/GradDescent.jl))
 - `atfrequency` : Choose how many variational parameters iterations are between hyperparameters optimization
 - `mean` : PriorMean object, check the documentation on it [`MeanPrior`](@ref meanprior)
 - `IndependentPriors` : Flag for setting independent or shared parameters among latent GPs
 - `optimizer` : Optimizer for inducing point locations (to be selected from [GradDescent.jl](https://github.com/jacobcvt12/GradDescent.jl))
 - `ArrayType` : Option for using different type of array for storage (allow for GPU usage)
"""
mutable struct MOARGP{T<:Real,TLikelihood<:Likelihood{T},TInference<:Inference,N,Q} <: AbstractGP{T,TLikelihood,TInference,N}
    X::Vector{Matrix{T}} #Feature vectors
    y::Vector #Output (-1,1 for classification, real for regression, matrix for multiclass)
    nSamples::Vector{Int64} # Number of data points
    nDim::Vector{Int64} # Number of covariates per data point
    nLatent::Int64 # Number of latent GPs
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



function MOARGP(
            X::AbstractVector{<:AbstractArray{T}},y::AbstractVector{<:AbstractVector},kernel::Kernel,
            likelihood::TLikelihood,inference::TInference,nLatent::Int,nInducingPoints::Int;
            verbose::Int=0,optimizer=Flux.ADAM(0.01),atfrequency::Int=1,
            mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(), variance::Real = 1.0,Aoptimizer=Flux.ADAM(0.01),
            Zoptimizer=false,
            ArrayType::UnionAll=Vector) where {T<:Real,TLikelihood<:Likelihood,TInference<:Inference}

            @assert length(y) > 0 "y should not be an empty vector"
            @assert length(X) == length(y)
            nTask = length(y)
            likelihoods = [deepcopy(likelihood) for _ in 1:nTask]
            nf_per_task = zeros(Int64,nTask)
            corrected_y = Vector(undef,nTask)
            for i in 1:nTask
                X[i],corrected_y[i],nf_per_task[i],likelihoods[i] = check_data!(X[i],y[i],likelihoods[i])
            end
            @assert check_implementation(:SVGP,likelihoods[1],inference) "The $likelihood is not compatible or implemented with the $inference"

            nSamples = size.(X,1); nDim = size.(X,2);
            if isa(optimizer,Bool)
                optimizer = optimizer ?  Flux.ADAM(0.01) : nothing
            end
            if isa(AOptimizer,Bool)
                Aoptimizer = Aoptimizer ? Flux.ADAM(0.01) : nothing
            end

            @assert nInducingPoints > 0 "The number of inducing points is incorrect (negative or bigger than number of samples)"
            nInducingPoints = nInducingPoints*ones(Int64,nLatent)
            if !all(nInducingPoints .< nSamples)
                @warn "Number of inducing points bigger than the number of points : reducing it to the number of samples: $(nSamples)"
                nInducingPoints .= ifelse.(nInducingPoints.>nSamples,nSamples,nInducingPoints)
            end
            Z = Vector{Matrix{T}}(undef,nLatent)
            for i in 1:nLatent
                if nInducingPoints[i] == nSamples[i]
                    Z[i] = X[i]
                else
                    Z[i] = KMeansInducingPoints(X[i],nInducingPoints[i],nMarkov=10)
                end
            end
            if isa(Zoptimizer,Bool)
                Zoptimizer = Zoptimizer ? Flux.ADAM(0.01) : nothing
            end
            Z = InducingPoints.(Z,[Zoptimizer])

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

            latent_f = ntuple( i -> _SVGP{T}(nFeatures[i],nMinibatch[i],Z[i],kernel,mean,variance,optimizer),nLatent)

            dpos = Normal(0.5,0.5)
            dneg = Normal(-0.5,0.5)
            A = zeros(T,nTask,nf_per_task[1],nLatent)
            for i in eachindex(A)
                p = rand(0:1)
                A[i] =  rand(Normal(0.0,1.0))#p*rand(dpos) + (1-p)*rand(dneg)
            end

            likelihoods .= init_likelihood.(likelihoods,inference,nf_per_task,nMinibatch,nFeatures)
            inference = tuple_inference(inference,nLatent,nFeatures[1],nSamples[1],nMinibatch[1]) #TODO hardcoded
            inference.xview = view.(X,[1:nMinibatch[1]],:)
            inference.yview = view(y,:)

            model = MOARGP{T,TLikelihood,typeof(inference),nTask,nLatent}(X,corrected_y,
                    nSamples, nDim, nLatent,
                    nTask, nf_per_task,
                    latent_f,likelihoods,inference,A,Aoptimizer,
                    verbose,atfrequency,false)
            # if isa(inference.optimizer,ALRSVI)
                # init!(model.inference,model)
            # end
            # return model
end

function Base.show(io::IO,model::MOARGP{T,<:Likelihood,<:Inference}) where {T}
    print(io,"Multioutput Autoregressive Gaussian Process with the likelihoods $(model.likelihood) infered by $(model.inference) ")
end

@traitimpl IsMultiOutput{MOARGP}
