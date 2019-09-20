"""
**GibbsSampling**

Draw samples from the true posterior via Gibbs Sampling.

```julia
GibbsSampling(;ϵ::T=1e-5,nBurnin::Int=100,samplefrequency::Int=10)
```
**Keywords arguments**

    - `ϵ::T` : convergence criteria
    - `nBurnin::Int` : Number of samples discarded before starting to save samples
    - `samplefrequency::Int` : Frequency of sampling
"""
mutable struct GibbsSampling{T<:Real} <: Inference{T}
    nBurnin::Integer
    samplefrequency::Integer
    ϵ::T #Convergence criteria
    nIter::Integer #Number of steps performed
    Stochastic::Bool #Use of mini-batches
    nSamples::Int64 # Number of samples
    nSamplesUsed::Int64 #Size of mini-batches
    MBIndices::AbstractVector #Indices of the minibatch
    ρ::T #Stochastic Coefficient
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
    sample_store::AbstractVector{AbstractVector{AbstractVector{T}}}
    x::SubArray{T,2,Matrix{T}}#,Tuple{Base.Slice{Base.OneTo{Int64}},Base.Slice{Base.OneTo{Int64}}},true}
    y::LatentArray{SubArray}
    function GibbsSampling{T}(nBurnin::Int,samplefrequency::Int,ϵ::T,nIter::Integer,Stochastic::Bool,nSamples::Integer,nSamplesUsed::Integer,MBIndices::AbstractVector,ρ::T,flag::Bool) where T
        return new{T}(nBurnin,samplefrequency,ϵ,nIter,Stochastic,nSamples,nSamplesUsed,MBIndices,ρ,flag)
    end
end

function GibbsSampling(;ϵ::T=1e-5,nBurnin::Int=100,samplefrequency::Int=10) where {T<:Real}
    GibbsSampling{Float64}(nBurnin,samplefrequency,ϵ,0,false,1,1,[1],1.0,true)
end

function Base.show(io::IO,inference::GibbsSampling{T}) where {T<:Real}
    print(io,"Gibbs Sampler")
end

function init_inference(inference::GibbsSampling{T},nLatent::Integer,nFeatures::Integer,nSamples::Integer,nSamplesUsed::Integer) where {T<:Real}
    inference.nSamples = nSamples
    inference.nSamplesUsed = nSamples
    inference.MBIndices = collect(1:nSamples)
    inference.sample_store = [Vector{Vector{T}}() for _ in 1:nLatent]
    return inference
end

function variational_updates!(model::VGP{T,L,GibbsSampling{T}}) where {T,L}
    sample_local!(model)
    sample_global!(model)
    if model.inference.nIter > model.inference.nBurnin && (model.inference.nIter-model.inference.nBurnin)%model.inference.samplefrequency==0
        for k in 1:model.nLatent
            push!(model.inference.sample_store[k],model.μ[k])
        end
    end
end

function ELBO(model::AbstractGP{T,<:Likelihood,<:GibbsSampling}) where {T}
    return NaN
end

function sample_global!(model::VGP{T,<:Likelihood,<:GibbsSampling}) where {T}
    model.Σ .= inv.(Symmetric.(Diagonal.(2.0.*∇E_Σ(model)).+model.invKnn)) # Definition of covariance for full conditional
    model.μ .= rand.(MvNormal.(model.Σ.*(∇E_μ(model).+model.invKnn.*model.μ₀),model.Σ)) # Sampling of f
    return nothing
end

function post_process!(model::AbstractGP{T,<:Likelihood,<:GibbsSampling}) where {T}
    for k in 1:model.nLatent
        model.μ[k] = vec(mean(hcat(model.inference.sample_store[k]...),dims=2))
        model.Σ[k] = Symmetric(cov(hcat(model.inference.sample_store[k]...),dims=2))
    end
    nothing
end
