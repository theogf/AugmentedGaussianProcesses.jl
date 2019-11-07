"""
```julia
HMCSampling(;ϵ::T=1e-5,nBurnin::Int=100,samplefrequency::Int=10)
```

Draw samples from the true posterior via Gibbs Sampling.

**Keywords arguments**
    - `ϵ::T` : convergence criteria
    - `nBurnin::Int` : Number of samples discarded before starting to save samples
    - `samplefrequency::Int` : Frequency of sampling
"""
mutable struct HMCSampling{T<:Real} <: SamplingInference{T}
    nBurnin::Integer # Number of burnin samples
    samplefrequency::Integer # Frequency at which samples are saved
    ϵ::T #Convergence criteria
    nIter::Integer #Number of steps performed
    Stochastic::Bool #Use of mini-batches
    nSamples::Int64 # Number of samples
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    sample_store::AbstractArray{T,3}
    x::SubArray{T,2,Matrix{T}}#,Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}}},true}
    y::LatentArray{SubArray}
    function HMCSampling{T}(nBurnin::Int,samplefrequency::Int,ϵ::T,nIter::Integer,Stochastic::Bool,nSamples::Integer,nSamplesUsed::Integer,MBIndices::AbstractVector,ρ::T,flag::Bool) where T
        return new{T}(nBurnin,samplefrequency,ϵ,nIter,Stochastic,nSamples,nSamplesUsed,MBIndices,ρ,flag)
    end
end

function HMCSampling(;ϵ::T=1e-5,nBurnin::Int=100,samplefrequency::Int=10) where {T<:Real}
    HMCSampling{Float64}(nBurnin,samplefrequency,ϵ,0,false,1,1,[1],1.0,true)
end

function Base.show(io::IO,inference::HMCSampling{T}) where {T<:Real}
    print(io,"Gibbs Sampler")
end

function grad_logpdf(model::MCGP{T,L,HMCSampling{T}},x) where {T,L}
    grad_log_likelihood(model.likelihood,get_y(model),x) + sum(grad_log_gp_prior.(model.f,x))
end

function init_sampler(inference::HMCSampling{T},nLatent::Integer,nFeatures::Integer,nSamples::Integer) where {T<:Real}
    inference.nSamples =
    inference.MBIndices = collect(1:nSamples)
    inference.sample_store = zeros(T,nLatent,nFeatures,n)
    return inference
end

function sample_parameters!(model::MCGP{T,L,HMCSampling{T}},nSamples::Int,callback) where {T,L}
    f_init = vcat(get_f(model)...)
    logpdf(x) = log_joint_model(model,x)
    gradlogpdf(x) = (log_joint_model(model,x),grad_log_joint_model(model,x))
    metric = i.metric
    h = Hamiltonian(metric, logpdf, gradlogpdf)
    int = Leapfrog(find_good_eps(h, f_init))
    prop = NUTS{MultinomialTS,GeneralisedNoUTurn}(int)
    adaptor = StanHMCAdaptor(n_adapts, Preconditioner(metric), NesterovDualAveraging(0.8, int.ϵ))
    samples, stats = sample(h, prop, f_init, nSamples, adaptor, n_adapts; progress=true)
end

function store_variables!(i::SamplingInference{T},fs)
    i.sample_store[:,:,(i.nIter-i.nBurnin)÷i.samplefrequency] .= hcat(fs...)
end

function post_process!(model::AbstractGP{T,<:Likelihood,<:HMCSampling}) where {T}
    for k in 1:model.nLatent
        model.μ[k] = vec(mean(hcat(model.inference.sample_store[k]...),dims=2))
        model.Σ[k] = Symmetric(cov(hcat(model.inference.sample_store[k]...),dims=2))
    end
    nothing
end
