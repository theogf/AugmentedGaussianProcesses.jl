"""
```julia
GaussianLikelihood(σ²::T=1e-3) #σ² is the variance
```
Gaussian noise :
```math
    p(y|f) = N(y|f,σ²)
```
There is no augmentation needed for this likelihood which is already conjugate to a Gaussian prior
"""
mutable struct GaussianLikelihood{T<:Real} <: RegressionLikelihood{T}
    σ²::T
    opt_noise::Bool
    θ::Vector{T}
    function GaussianLikelihood{T}(σ²::T,opt_noise::Bool) where {T<:Real}
        new{T}(σ²,opt_noise)
    end
    function GaussianLikelihood{T}(σ²::T,opt_noise::Bool,θ::AbstractVector{T}) where {T<:Real}
        new{T}(σ²,opt_noise,θ)
    end
end

function GaussianLikelihood(σ²::T=1e-3;opt_noise::Bool=true) where {T<:Real}
    GaussianLikelihood{T}(σ²,opt_noise)
end

function pdf(l::GaussianLikelihood,y::Real,f::Real)
    pdf(Normal(y,l.σ²),f)
end

function logpdf(l::GaussianLikelihood,y::Real,f::Real)
    logpdf(Normal(y,l.σ²),f)
end

export noise

noise(l::GaussianLikelihood) = l.σ²

function Base.show(io::IO,model::GaussianLikelihood{T}) where T
    print(io,"Gaussian likelihood")
end

function compute_proba(l::GaussianLikelihood{T},μ::AbstractVector{T},σ²::AbstractVector{T}) where {T<:Real}
    return μ,max.(σ²,zero(σ²)).+ l.σ²
end

function init_likelihood(likelihood::GaussianLikelihood{T},inference::Inference{T},nLatent::Int,nSamplesUsed::Int,nFeatures::Int) where {T<:Real}
    return GaussianLikelihood{T}(likelihood.σ²,likelihood.opt_noise,fill(inv(likelihood.σ²),nSamplesUsed))
end

function local_updates!(l::GaussianLikelihood{T},y::AbstractVector,μ::AbstractVector,Σ::AbstractVector) where {T}
    if l.opt_noise
        # ρ = inv(sqrt(1+model.inference.nIter))
        l.σ² = sum(abs2.(y-μ)+Σ)/length(y)
    end
    l.θ .= inv(l.σ²)
end

@inline ∇E_μ(l::GaussianLikelihood{T},::AOptimizer,y::AbstractVector) where {T} = (y./l.σ²,)
@inline ∇E_Σ(l::GaussianLikelihood{T},::AOptimizer,y::AbstractVector) where {T} = (0.5*l.θ,)

### Special case where the ELBO is equal to the marginal likelihood
function ELBO(model::GP{T,GaussianLikelihood{T}}) where {T}
    return -0.5*sum(broadcast((y,invK)->dot(y,invK*y) - logdet(invK)+ model.nFeatures*log(twoπ),model.y,model.invKnn))
end

function expec_logpdf(l::GaussianLikelihood{T},i::AnalyticVI,y::AbstractVector,μ::AbstractVector,diag_cov::AbstractVector) where {T}
    return -0.5*i.ρ*(length(y)*(log(twoπ)+log(l.σ²))+sum(abs2.(y-μ)+diag_cov)/l.σ²)
end

AugmentedKL(::GaussianLikelihood{T},::AbstractVector) where {T} = zero(T)
