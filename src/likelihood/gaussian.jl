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
    opt_noise
    θ::Vector{T}
    function GaussianLikelihood{T}(σ²::T,opt_noise) where {T<:Real}
        new{T}(σ²,opt_noise)
    end
    function GaussianLikelihood{T}(σ²::T,opt_noise,θ::AbstractVector{T}) where {T<:Real}
        new{T}(σ²,opt_noise,θ)
    end
end

function GaussianLikelihood(σ²::T=1e-3;opt_noise=false) where {T<:Real}
    if isa(opt_noise,Bool)
        opt_noise = opt_noise ? ADAM(0.05) : nothing
    end
    GaussianLikelihood{T}(σ²,opt_noise)
end

function pdf(l::GaussianLikelihood,y::Real,f::Real)
    Distributions.pdf(Normal(y,l.σ²),f)
end

function logpdf(l::GaussianLikelihood,y::Real,f::Real)
    Distributions.logpdf(Normal(y,l.σ²),f)
end

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

function local_updates!(l::GaussianLikelihood{T},y::AbstractVector,μ::AbstractVector,diag_cov_f::AbstractVector) where {T}
    if !isnothing(l.opt_noise)
        if l.opt_noise.t<=10
            # update(l.opt_noise,zero(T))
        else
            l.σ² = exp(log(l.σ²)+Flux.Optimise.apply!(l.opt_noise,l.σ²,0.5*((sum(abs2,y-μ)+sum(diag_cov_f))/l.σ²-length(y))))
        end
    end
    l.θ .= inv(l.σ²)
end

@inline ∇E_μ(l::GaussianLikelihood{T},::AOptimizer,y::AbstractVector) where {T} = (y./l.σ²,)
@inline ∇E_Σ(l::GaussianLikelihood{T},::AOptimizer,y::AbstractVector) where {T} = (0.5*l.θ,)

function expec_log_likelihood(l::GaussianLikelihood{T},i::AnalyticVI,y::AbstractVector,μ::AbstractVector,diag_cov::AbstractVector) where {T}
    return -0.5*i.ρ*(length(y)*(log(twoπ)+log(l.σ²))+sum(abs2.(y-μ)+diag_cov)/l.σ²)
end

AugmentedKL(::GaussianLikelihood{T},::AbstractVector) where {T} = zero(T)
