"""
**SoftMax Likelihood**

Multiclass likelihood with Softmax transformation: ``p(y=i|{f_k}) = \\exp(f_i)/ \\sum_{j=1}\\exp(f_j) ``

There is no possible augmentation for this likelihood
"""
struct SoftMaxLikelihood{T<:Real} <: MultiClassLikelihood{T}
    Y::AbstractVector{BitVector} #Mapping from instances to classes
    class_mapping::AbstractVector{Any} # Classes labels mapping
    ind_mapping::Dict{Any,Int} # Mapping from label to index
    y_class::AbstractVector{Int64} #GP Index for each sample
    θ::AbstractVector{AbstractVector{T}} # Variational parameter of Polya-Gamma distribution
    function SoftMaxLikelihood{T}() where {T<:Real}
        new{T}()
    end
    function SoftMaxLikelihood{T}(Y::AbstractVector{<:BitVector},
    class_mapping::AbstractVector, ind_mapping::Dict{<:Any,<:Int},y_class::AbstractVector{<:Int}) where {T<:Real}
        new{T}(Y,class_mapping,ind_mapping,y_class)
    end
end

function SoftMaxLikelihood()
    SoftMaxLikelihood{Float64}()
end

function pdf(l::SoftMaxLikelihood,f::AbstractVector)
    StatsFuns.softmax(f)
end

function pdf(l::SoftMaxLikelihood,y::Int,f::AbstractVector{<:Real})
    StatsFuns.softmax(f)[y]
end

function Base.show(io::IO,model::SoftMaxLikelihood{T}) where T
    print(io,"Softmax likelihood")
end


function init_likelihood(likelihood::SoftMaxLikelihood{T},nLatent::Integer,nSamplesUsed::Integer) where T
    if inference isa GibbsSampling
        θ = [abs.(rand(T,nSamplesUsed))*2 for i in 1:nLatent]
        LogisticSoftMaxLikelihood{T}(likelihood.Y,likelihood.class_mapping,likelihood.ind_mapping,likelihood.y_class,θ)
    else
        return likelihood
    end
end

function sample_local!(model::VGP{T,<:SoftMaxLikelihood,<:GibbsSampling}) where {T}
    pg = PolyaGammaDist()
    model.likelihood.θ .= broadcast((y::BitVector,γ::AbstractVector{<:Real},μ::AbstractVector{<:Real},i::Int64)->draw.([pg],1.0,μ-logsumexp()),model.likelihood.Y,model.likelihood.γ,model.μ,1:model.nLatent)
    return nothing #TODO FINISH AT SOME POINT
end

function grad_samples(model::AbstractGP{T,<:SoftMaxLikelihood},samples::AbstractMatrix{T},index::Integer) where {T}
    class = model.likelihood.y_class[index]
    grad_μ = zeros(T,model.nLatent)
    grad_Σ = zeros(T,model.nLatent)
    nSamples = size(samples,1)
    samples .= mapslices(StatsFuns.softmax,samples,dims=2)
    t = 0.0
    @inbounds for i in 1:nSamples
        s = samples[i,class]::T
        @views g_μ = grad_softmax(samples[i,:],class)/s
        grad_μ += g_μ
        @views h = diaghessian_softmax(samples[i,:],class)/s
        grad_Σ += h - abs2.(g_μ)
    end
    for k in 1:model.nLatent
        model.inference.∇μE[k][index] = grad_μ[k]/nSamples
        model.inference.∇ΣE[k][index] = 0.5.*grad_Σ[k]/nSamples
    end
end

function log_like_samples(model::AbstractGP{T,<:SoftMaxLikelihood},samples::AbstractMatrix,index::Integer) where {T}
    class = model.likelihood.y_class[index]
    nSamples = size(samples,1)
    loglike = mapslices(logsumexp,samples,dims=2)/nSamples
end

function grad_softmax(s::AbstractVector{<:Real},i::Integer)
    return (δ.(i,eachindex(s))-s)*s[i]
end

function diaghessian_softmax(s::AbstractVector{<:Real},i::Integer)
    return s[i]*(abs2.(δ.(i,eachindex(s))-s) - s.*(1.0.-s))
end

function hessian_softmax(s::AbstractVector{<:Real},i::Integer)
    m = length(s)
    hessian = zeros(m,m)
    for j in 1:m
        for k in 1:m
            hessian[j,k] = s[i]*((δ(i,k)-s[k])*(δ(i,j)-s[j])-s[j]*(δ(j,k)-s[k]))
        end
    end
    return hessian
end
