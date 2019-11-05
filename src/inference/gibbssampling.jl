"""
```julia
GibbsSampling(;ϵ::T=1e-5,nBurnin::Int=100,samplefrequency::Int=10)
```

Draw samples from the true posterior via Gibbs Sampling.

**Keywords arguments**
    - `ϵ::T` : convergence criteria
    - `nBurnin::Int` : Number of samples discarded before starting to save samples
    - `samplefrequency::Int` : Frequency of sampling
"""
mutable struct GibbsSampling{T<:Real} <: SamplingInference{T}
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

function init_sampler(inference::GibbsSampling{T},nLatent::Integer,nFeatures::Integer,nSamples::Integer) where {T<:Real}
    inference.nSamples =
    inference.MBIndices = collect(1:nSamples)
    inference.sample_store = zeros(T,nLatent,nFeatures,n)
    return inference
end

<<<<<<< HEAD
function sample_parameters!(model::AbstractGP{T,L,GibbsSampling{T}}) where {T,L}
    sample_local!(model)
    sample_global!(model)
    if model.inference.nIter > model.inference.nBurnin && (model.inference.nIter-model.inference.nBurnin)%model.inference.samplefrequency==0
        store_variables!(model)
=======
function sample_parameters!(model::GPMC{T,L,GibbsSampling{T}},nSamples::Int,callback) where {T,L}
    model.inference.sample_store = zeros(T,model.nLatent,model.nFeatures,nSamples)
    computeMatrices!(model)
    for i in 1:(model.inference.nBurnin+nSamples*model.inference.atfrequency)
        sample_local!(model.likelihood,get_y(model),get_f(model))
        sample_global!.(∇E_μ(model.likelihood,model.inference.vi_opt[1],get_y(model)),
        ∇E_Σ(model.likelihood,model.inference.vi_opt[1],get_y(model)),model.f)
        if model.inference.nIter > model.inference.nBurnin && (model.inference.nIter-model.inference.nBurnin)%model.inference.samplefrequency==0
            store_variables!(model.inference,get_f(model))
        end
>>>>>>> dbea8df91f33794d97b6ab471444b4dcfbeee5ab
    end
    symbols = [Symbol(f,"_",i,"_",j) for j in 1:model.nFeatures, i in 1:model.nLatent]
    Chain(reshape(model.inference.sample_store,model.nLatent*model.nFeatures,:,1),symbols)
end

sample_local!(l::Likelihood,y,μ::Tuple{<:AbstractVector{T}},Σ::Tuple{<:AbstractVector{T}}) where {T} =sample_local!(l,y,first(μ),first(Σ))
set_ω!(l::Likelihood,ω) = l.θ .= ω
get_ω(l::Likelihood) = l.θ

function logpdf(model::AbstractGP{T,<:Likelihood,<:GibbsSampling}) where {T}
    return 0.0
end

function sample_global!(∇E_Σ::AbstractVector,∇E_μ::AbstractVector,gp::_GPMC{T}) where {T}
    Σ = inv(Diagonal(2.0*∇E_Σ)+inv(gp.K))
    gp.f .= rand(MvNormal(Σ*(∇E_μ+gp.K/gp.μ₀),Σ))
    return nothing
end

function store_variables!(i::SamplingInference{T},fs)
    i.sample_store[:,:,(i.nIter-i.nBurnin)÷i.samplefrequency] .= hcat(fs...)
end

function post_process!(model::AbstractGP{T,<:Likelihood,<:GibbsSampling}) where {T}
    for k in 1:model.nLatent
        model.μ[k] = vec(mean(hcat(model.inference.sample_store[k]...),dims=2))
        model.Σ[k] = Symmetric(cov(hcat(model.inference.sample_store[k]...),dims=2))
    end
    nothing
end
