"""
```julia
    NegBinomialLikelihood(r::Int=10)
```

[Negative Binomial likelihood](https://en.wikipedia.org/wiki/Negative_binomial_distribution) with number of failures `r`
```math
    p(y|r,f) = binomial(y+r-1,y) (1-σ(f))ʳσ(f)ʸ
```
Where `σ` is the logistic function

"""
struct NegBinomialLikelihood{T<:Real} <: EventLikelihood{T}
    r::Int
    c::Vector{T}
    θ::Vector{T}
    function NegBinomialLikelihood{T}(r::Int) where {T<:Real}
        new{T}(r)
    end
    function NegBinomialLikelihood{T}(r,c,θ) where {T<:Real}
        new{T}(r,c,θ)
    end
end

function NegBinomialLikelihood(r::Int=10) where {T<:Real}
    NegBinomialLikelihood{Float64}(r)
end

function init_likelihood(likelihood::NegBinomialLikelihood{T},inference::Inference{T},nLatent::Integer,nSamplesUsed::Int,nFeatures::Int) where T
    NegBinomialLikelihood{T}(
    likelihood.r,
    rand(T,nSamplesUsed),
    zeros(T,nSamplesUsed))
end

function pdf(l::NegBinomialLikelihood,y::Real,f::Real)
    pdf(NegativeBinomial(lr,get_p(l,f)),y)
end

function expec_count(l::NegBinomialLikelihood,f)
    broadcast((p,r)->p*r./(1.0.-p) ,get_p.(l,f),l.r)
end

function get_p(::NegBinomialLikelihood,f)
    logistic.(f)
end

function Base.show(io::IO,model::NegBinomialLikelihood{T}) where T
    print(io,"Negative Binomial Likelihood")
end

function compute_proba(l::NegBinomialLikelihood{T},μ::Vector{T},σ²::Vector{T}) where {T<:Real}
    N = length(μ)
    pred = zeros(T,N)
    for i in 1:N
        x = pred_nodes.*sqrt(max(σ²[i],zero(T))).+μ[i]
        pred[i] = dot(pred_weights,get_p.(l,x))
    end
    return pred
end

## Local Updates ##

function local_updates!(l::NegBinomialLikelihood{T},y::AbstractVector,μ::AbstractVector,Σ::AbstractVector) where {T}
    l.c .= sqrt.(abs2.(μ) + Σ)
    l.θ .= (l.r.+y)./l.c.*tanh.(0.5*l.c)
end

## Global Updates ##

@inline ∇E_μ(l::NegBinomialLikelihood{T},::AOptimizer,y::AbstractVector) where {T} = (0.5*(y.-l.r),)
@inline ∇E_Σ(l::NegBinomialLikelihood{T},::AOptimizer,y::AbstractVector) where {T} = (0.5.*l.θ,)

## ELBO Section ##

AugmentedKL(l::NegBinomialLikelihood{T},y::AbstractVector) where {T} = PolyaGammaKL(l,y)

function logabsbinomial(n,k)
    log(binomial(n,k))
end

function expec_log_likelihood(l::NegBinomialLikelihood{T},i::AnalyticVI,y::AbstractVector,μ::AbstractVector,diag_cov::AbstractVector) where {T}
    tot = sum(logabsbinomial.(y.+(l.r-1),y))-log(2.0)*sum((y.+l.r))
    tot += 0.5*dot(μ,(y.-l.r))-0.5*dot(l.θ,μ)-0.5*dot(l.θ,diag_cov)
    return tot
end

function PolyaGammaKL(l::NegBinomialLikelihood,y::AbstractVector) where {T}
    PolyaGammaKL(y.+l.r,l.c,l.θ)
end
