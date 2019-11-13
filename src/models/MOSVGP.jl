"""
Class for multi-output sparse variational Gaussian Processes

```julia
MOSVGP(X::AbstractArray{T1},y::AbstractVector{AbstractArray{T2}},kernel::Kernel,
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
mutable struct MOSVGP{T<:Real,TLikelihood<:Likelihood{T},TInference<:Inference,TGP<:Abstract_GP{T},N,Q} <: AbstractGP{T,TLikelihood,TInference,TGP,N}
    X::Matrix{T} #Feature vectors
    y::Vector #Output (-1,1 for classification, real for regression, matrix for multiclass)
    nSamples::Int64 # Number of data points
    nDim::Int64 # Number of covariates per data point
    nFeatures::Int64 # Number of features of the GP (equal to number of points)
    nLatent::Int64 # Number of latent GPs
    nTask::Int64
    nf_per_task::Vector{Int64}
    f::NTuple{Q,TGP}
    likelihood::Vector{TLikelihood}
    inference::TInference
    A::Array{T,3}
    verbose::Int64
    atfrequency::Int64
    Trained::Bool
end



function MOSVGP(
            X::AbstractArray{T1},y::AbstractVector{<:AbstractVector{T2}},kernel::Kernel,
            likelihood::TLikelihood,inference::TInference,nLatent::Int,nInducingPoints::Int;
            verbose::Int=0,optimizer::Union{Optimizer,Nothing,Bool}=Adam(α=0.01),atfrequency::Int=1,
            mean::Union{<:Real,AbstractVector{<:Real},PriorMean}=ZeroMean(), variance::Real = 1.0,
            Zoptimizer::Union{Optimizer,Nothing,Bool}=false,
            ArrayType::UnionAll=Vector) where {T1<:Real,T2,TLikelihood<:Likelihood,TInference<:Inference}

            @assert length(y) > 0 "y should not be an empty vector"
            nTask = length(y)
            likelihoods = [deepcopy(likelihood) for _ in 1:nTask]
            nf_per_task = zeros(Int64,nTask)
            for i in 1:nTask
                X,y[i],nf_per_task[i],likelihoods[i] = check_data!(X,y[i],likelihoods[i])
            end
            @assert check_implementation(:SVGP,likelihoods[1],inference) "The $likelihood is not compatible or implemented with the $inference"

            nSamples = size(X,1); nDim = size(X,2);
            if isa(optimizer,Bool)
                optimizer = optimizer ? Adam(α=0.01) : nothing
            end

            @assert nInducingPoints > 0 && nInducingPoints < nSamples "The number of inducing points is incorrect (negative or bigger than number of samples)"
            Z = KMeansInducingPoints(X,nInducingPoints,nMarkov=10)
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
                @assert inference.nMinibatch > 0 && inference.nMinibatch < nSample "The size of mini-batch $(inference.nMinibatch) is incorrect (negative or bigger than number of samples), please set nMinibatch correctly in the inference object"
                nMinibatch = inference.nMinibatch
            end

            latent_f = ntuple( _ -> _SVGP{T1}(nFeatures,nMinibatch,Z,kernel,mean,variance,optimizer),nLatent)

            dpos = Normal(0.5,0.5)
            dneg = Normal(-0.5,0.5)
            A = zeros(T1,nTask,nf_per_task[1],nLatent)
            for i in eachindex(A)
                p = rand(0:1)
                A[i] =  rand(Normal(0.0,1.0))#p*rand(dpos) + (1-p)*rand(dneg)
            end

            likelihoods .= init_likelihood.(likelihoods,inference,nLatent,nMinibatch,nFeatures)
            inference = tuple_inference(inference,nLatent,nFeatures,nSamples,nMinibatch)
            inference.xview = view(X,1:nMinibatch,:)
            inference.yview = view(y,:)

            model = MOSVGP{T1,TLikelihood,typeof(inference),_SVGP{T1},nTask,nLatent}(X,y,
                    nSamples, nDim, nFeatures, nLatent,
                    nTask, nf_per_task,
                    latent_f,likelihoods,inference,A,
                    verbose,atfrequency,false)
            # if isa(inference.optimizer,ALRSVI)
                # init!(model.inference,model)
            # end
            # return model
end

function Base.show(io::IO,model::MOSVGP{T,<:Likelihood,<:Inference}) where {T}
    print(io,"Multioutput Sparse Variational Gaussian Process with the likelihoods $(model.likelihood) infered by $(model.inference) ")
end

function mean_f(model::MOSVGP{T}) where {T}
    μ = mean_f.(model.f)
    f = []
    for i in 1:model.nTask
        x = ntuple(_->zeros(T,model.inference.nMinibatch),model.nf_per_task[i])
        for j in 1:model.nf_per_task[i]
            x[j] .= sum(vec(model.A[i,j,:]) .* μ)
        end
        push!(f,x)
    end
    return f
end

function diag_cov_f(model::MOSVGP{T}) where {T}
    Σ = diag_cov_f.(model.f)
    cov_f = []
    for i in 1:model.nTask
        x = ntuple(_->zeros(T,model.inference.nMinibatch),model.nf_per_task[i])
        for j in 1:model.nf_per_task[i]
            x[j] .= sum(vec(model.A[i,j,:]).^2 .* Σ)
        end
        push!(cov_f,x)
    end
    return cov_f
end

get_y(model::MOSVGP) = view.(model.y,[model.inference.MBIndices])

## return the linear sum of the expectation gradient given μ ##
function ∇E_μ(model::MOSVGP{T}) where {T}
    ∇ = [zeros(T,model.inference.nMinibatch) for _ in 1:model.nLatent]
    ∇Eμs = ∇E_μ.(model.likelihood,model.inference.vi_opt[1:1],get_y(model))
    ∇EΣs = ∇E_Σ.(model.likelihood,model.inference.vi_opt[1:1],get_y(model))
    μs = mean_f.(model.f)
    for t in 1:model.nTask
        for j in 1:model.nf_per_task[t]
            for q in 1:model.nLatent
                ∇[q] .+= model.A[t,j,q] * (∇Eμs[t][j]  - ∇EΣs[t][j].*sum(model.A[t,j,qq]*μs[qq] for qq in 1:model.nLatent if qq!=q))
            end
        end
    end
    return ∇
end

## return the linear sum of the expectation gradient given diag(Σ) ##
function ∇E_Σ(model::MOSVGP{T}) where {T}
    ∇ = [zeros(T,model.inference.nMinibatch) for _ in 1:model.nLatent]
    ∇Es = ∇E_Σ.(model.likelihood,model.inference.vi_opt[1:1],get_y(model))
    for t in 1:model.nTask
        for j in 1:model.nf_per_task[t]
            for q in 1:model.nLatent
                ∇[q] .+= model.A[t,j,q]^2 * ∇Es[t][j]
            end
        end
    end
    return ∇
end

get_Z(model::MOSVGP) = getproperty.(getproperty.(model.f,:Z),:Z)


function update_A!(model::MOSVGP)
    μ = mean_f.(model.f) # κμ
    Σ = diag_cov_f.(model.f) #K̃ + κΣκ
    ∇Eμ = ∇E_μ.(model.likelihood,model.inference.vi_opt[1:1],get_y(model))
    ∇EΣ = ∇E_Σ.(model.likelihood,model.inference.vi_opt[1:1],get_y(model))
    new_A = zero(model.A)
    for t in 1:model.nTask
        for j in 1:model.nf_per_task[t]
            for q in 1:model.nLatent
                x1 = dot(∇Eμ[t][j],μ[q])-dot(∇EΣ[t][j],μ[q].*sum(model.A[t,j,qq]*μ[qq] for qq in 1:model.nLatent if qq!=q))
                x2 = dot(∇EΣ[t][j],abs2.(μ[q])+Σ[q])
                new_A[t,j,q] = x1/(2*x2)
            end
            # model.A[t,j,:]./=sum(model.A[t,j,:])
        end
    end
    # model.A .= new_A
end

function ELBO(model::MOSVGP{T}) where {T}
    tot = zero(T)
    tot += model.inference.ρ*sum(expec_logpdf.(model.likelihood,model.inference,get_y(model),mean_f(model),diag_cov_f(model)))
    tot -= GaussianKL(model)
    tot -= model.inference.ρ*sum(AugmentedKL.(model.likelihood,get_y(model)))
end
